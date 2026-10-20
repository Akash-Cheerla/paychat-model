"""Generate (selected lines -> written recap) pairs with an open-weight teacher.

For the conversation recap. The rules can already pick the right lines out of a window -
12 of the 33 in a work room, 12 of 26 in a flatmates room, with no fabrication because
every line is something somebody typed. What they cannot do is write a recap: five
attempts at grouping and titling with rules produced "Too", "Saturday Kitchen" and
"I'll Plumber", and the shipped student handed the same lines produced "owe 680 from
dinner" when 680 was the cab fare.

Akash, 2026-10-19: "cant keep 1-2 lines and for each small topic and just 1-3 points we
can have sumary for each, thats not professional" - the selected-lines version is a
record rather than a recap. It lists the conversation's steps where a person would write
its conclusion: four lines about a split where one would do.

So the division of labour is the same as the per-message card, one step further on:

    rules   decide WHICH lines matter, and compute the money
    teacher writes the recap from those lines
    student learns to do the writing on our own hardware

WHY THE RULES KEEP THE ARITHMETIC. "18000 split three ways so 6000 each" plus "800 for
electricity" is 6800 each, and a 250M seq2seq cannot be trusted to add - this is the
model that produced "owe 680 from dinner". app.py already computes shares and has 194/195
on tests/test_split_matrix.py, so a figure the rules computed is injected into the input
and the model only has to place it in a sentence. A number the model emits that is not in
its input is refused outright.

TEACHER IS OPEN WEIGHT, ON OUR OWN HARDWARE, NOT AN API. Two reasons and the second is
the one that matters: Akash's standing rule is that nothing sent to an API becomes
training data, and most providers' terms forbid training another model on their output -
a student trained that way would sit in production legally tainted.

BOTH HALVES COME FROM THE TEACHER, IN TWO CALLS. The window is training data as much as
the recap, so it cannot come from a different source. And asking for a conversation and
its recap together lets the model write a conversation to fit a recap it has already
planned, teaching the student a relationship that will not exist at inference.

    # dry run on CPU with a tiny model, to prove every cell works before Colab
    python data_gen/distill_recaps.py --n 2 --model Qwen/Qwen2.5-0.5B-Instruct \
        --out /tmp/recaps.jsonl --verbose
"""
import argparse
import io
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import room_summarizer as R   # noqa: E402

# The rooms this product is actually in. Deliberately not all work chat: the logs are a
# dev team, and a consumer app is mostly rent, rides, dinners and family.
SETTINGS = [
    "three flatmates discussing rent, a utility bill and a repair someone has to arrange",
    "a group of friends settling up after a dinner out, with one person short of cash",
    "a small product team on a release day: a build to test, a bug found, a date moving",
    "two people arranging a shared cab to the airport and an old debt between them",
    "a family group planning a weekend trip, who is driving and who pays for what",
    "colleagues deciding a meeting time across two cities, with one person travelling",
    "a group organising a birthday gift: who collects, how much each, by when",
    "two housemates and a landlord visit, plus the deposit still not returned",
    "a project group splitting a software subscription and arguing about the renewal",
    "friends planning a match ticket purchase, one books and the others transfer",
]

TOPIC_COUNTS = [2, 2, 3, 3, 3, 4]

# A worked example, because an instruction with six constraints is not followed from the
# instruction alone. The earlier round measured this: few-shot more than doubled what a
# 1.5B produced, and on a dry run Qwen2.5-0.5B answered the bare prompt with "Hello! It
# seems I didn't see your name when you made the previous message" rather than a chat.
WINDOW_EXAMPLE = """Example of the format and the texture wanted:

priya: rent's due on the 5th
sam: how much this time
priya: 18000, so 6000 each
sam: ok
nina: the geyser is leaking again
sam: since when
nina: two days, it's getting worse
priya: i'll call the plumber
sam: the same guy?
priya: yeah, he can come saturday
nina: i'm out saturday
priya: fine, i'll be here
sam: wait electricity came too
sam: 2400 for last month
priya: so 6800 each then
sam: ya
priya: plumber wants 600 for the visit
nina: 200 each
sam: lol this month is expensive
"""

WINDOW_PROMPT = """Write a realistic group chat, in English only.

Setting: {setting}
It must contain {topics} separate topics running at the same time, interleaved - people
reply about whichever topic they care about, not in neat blocks.

Rules for realism:
- {n} messages total, each on its own line as "name: message"
- most messages are SHORT, under ten words, the way people really type
- include filler that means nothing: "ok", "lol", "sure", "morning"
- someone corrects an earlier number or detail a few messages later
- at least one amount of money and at least one date or time
- no greetings-only messages at the start, get into it
- do not number the messages, do not add a summary, output only the chat

{example}
Now write a different chat for the setting above. Output the chat and nothing else."""


# The dry run has to prove MY pipeline - parse, select, compute, prompt, parse back -
# and it cannot do that while it depends on a tiny model choosing to comply. This chat
# stands in for the teacher so every cell downstream is exercised on CPU before Colab.
STUB_CHAT = """maya: dinner last night came to 3600 total
raj: how many of us were there
maya: six, so 600 each
raj: ok sending
dev: morning
maya: also the cab back was 480
raj: who paid that
dev: i did
maya: add it to the split then
dev: lol
raj: wait the demo moved
raj: it's wednesday 11am now instead of friday
maya: oh
dev: does that still work for you
raj: should do
maya: QA-204 is still failing on the new build
dev: that's mine, i'll look today
raj: please check it before wednesday
maya: ya
dev: the login screen is blank on android too
raj: let's sync at 3 tomorrow
maya: works
dev: cool"""

RECAP_PROMPT = """These are the lines worth remembering from a group chat, already
selected, in time order. Some carry a figure the app computed, marked [computed].

{points}

Write the recap someone would read a day later to catch up. Rules:

- Use exactly these sections, in this order, and omit any section with nothing in it:
  DECISIONS, ACTION ITEMS, OPEN QUESTIONS, INFORMATION
- One line per item, starting with "- "
- Name the owner when the lines name one: "Priya to call the plumber"
- State the CONCLUSION, not the steps. If the lines work out a split over several
  messages, give the final figure once, not the working.
- Every number must appear in the lines above. Never calculate a new one.
- A QUESTION stays a question. If a line asks "so we're paying 9,600 each?" that figure
  is not agreed - it belongs under OPEN QUESTIONS, never stated as a fact under
  INFORMATION or DECISIONS. Only state an amount as settled when a line settles it.
- Never write an item nobody said. If something is unclear, leave it out.
- Prefer the specific over the vague. Drop "this needs urgent attention" and "let's
  discuss further" - they say nothing a reader can act on.
- No preamble, no closing line, no heading of your own.

Output the recap and nothing else."""


def _load(model_id):
    """The teacher. Local weights, no network calls at inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    return tok, model, torch


def _ask(tok, model, torch, prompt, max_new=700):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=3072)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.9, top_p=0.95)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


_LINE = re.compile(r"^\s*([A-Za-z][\w .'-]{0,20}?)\s*:\s*(.+?)\s*$")


def parse_chat(text, cap=60):
    """"name: message" lines into the message dicts room_summarizer expects."""
    out = []
    for raw in (text or "").splitlines():
        m = _LINE.match(raw)
        if not m:
            continue
        sender, body = m.group(1).strip(), m.group(2).strip()
        if not body or len(sender.split()) > 3:
            continue
        if body.lower().startswith(("decisions", "action items", "open questions",
                                    "information", "summary", "recap")):
            continue
        out.append({"sender": sender, "text": body})
        if len(out) >= cap:
            break
    for i, m in enumerate(out):
        m["ts"] = f"2026-11-20T09:{i // 2:02d}:{(i * 7) % 60:02d}Z"
    return out


_AMOUNT = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d{1,2})?)(?![\w.])")
_WAYS = re.compile(r"\b(?:split|between|among|divided)\s+(?:by\s+)?"
                   r"(?:(\d+)|(two|three|four|five|six|seven|eight))\b"
                   r"|\b(?:(\d+)|(two|three|four|five|six|seven|eight))\s+(?:of us|ways|people)\b",
                   re.IGNORECASE)
_WORD_N = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def computed_share(points):
    """The per-person figure, worked out by rules rather than by the model.

    The teacher is told to state the conclusion of a split, and a 250M student cannot be
    trusted to add - it is the model that wrote "owe 680 from dinner" when 680 was the cab
    fare. So when the lines contain a total and a number of ways, the share is computed
    here and handed to the teacher as a given.
    """
    blob = " ".join(p["text"] for p in points)
    ways = None
    m = _WAYS.search(blob)
    if m:
        for g in m.groups():
            if not g:
                continue
            ways = int(g) if g.isdigit() else _WORD_N.get(g.lower())
            if ways:
                break
    if not ways or ways < 2:
        return None
    amounts = [float(a.replace(",", "")) for a in _AMOUNT.findall(blob)]
    amounts = [a for a in amounts if a >= ways * 10]      # a total, not a headcount
    if not amounts:
        return None
    total = max(amounts)
    share = total / ways
    if abs(share - round(share)) > 0.005:
        return f"{total:,.0f} split {ways} ways is {share:,.2f} each"
    return f"{total:,.0f} split {ways} ways is {round(share):,} each"


def one_pair(tok, model, torch, rnd, verbose=False, stub=False):
    setting = rnd.choice(SETTINGS)
    topics = rnd.choice(TOPIC_COUNTS)
    n = rnd.choice([22, 26, 30, 34, 38])
    if stub:
        chat = STUB_CHAT
    else:
        chat = _ask(tok, model, torch,
                    WINDOW_PROMPT.format(setting=setting, topics=topics, n=n,
                                         example=WINDOW_EXAMPLE), max_new=900)
    msgs = parse_chat(chat)
    if len(msgs) < R.MIN_MESSAGES:
        return None, "too few messages parsed"
    summary = R.summarize_room(R.window(msgs), room="distill")
    if not summary:
        return None, "rules found nothing worth a recap"

    lines = [f"- {p['text']}" for p in summary.points]
    share = computed_share(summary.points)
    if share:
        lines.append(f"- [computed] {share}")
    recap = _ask(tok, model, torch,
                 RECAP_PROMPT.format(points="\n".join(lines)), max_new=420)
    if verbose:
        print("\n--- chat ---")
        for m in msgs[:8]:
            print(f"   {m['sender']}: {m['text']}")
        print("--- selected ---")
        for line in lines:
            print(f"   {line}")
        print("--- recap ---")
        print("   " + recap.replace("\n", "\n   "))
    return {"setting": setting, "topics": topics,
            "messages": msgs,
            "selected": [p["text"] for p in summary.points],
            "computed": share,
            "input": "\n".join(lines),
            "target": recap}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--out", default="data/distill/recaps.jsonl")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stub", action="store_true",
                    help="use a canned chat, to prove the pipeline on CPU")
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    tok, model, torch = _load(a.model)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = {}
    with io.open(out, "a", encoding="utf-8") as f:
        for i in range(a.n):
            pair, why = one_pair(tok, model, torch, rnd, a.verbose, a.stub)
            if not pair:
                skipped[why] = skipped.get(why, 0) + 1
                print(f"  {i + 1}/{a.n}  skipped: {why}")
                continue
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            f.flush()
            kept += 1
            print(f"  {i + 1}/{a.n}  kept  ({len(pair['selected'])} lines selected)")
    print(f"\n  {kept} pairs -> {a.out}")
    if skipped:
        print(f"  skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
