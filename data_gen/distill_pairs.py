"""Generate (chat message -> summary card) training pairs with an open-weight teacher.

For distilling the Summary feature into a small student. Measured 2026-10-09, every stock
small model fails in one of two directions - the rules keep 76.7% of what matters but
60.7% of their bullets are verbatim copies, while small models rewrite properly and keep
only 13-49%. A teacher does both (80.7% kept, 1% verbatim), so the student learns from the
teacher rather than from a prompt.

The teacher is an OPEN-WEIGHT model run on our own hardware, not an API. Two reasons, and
the second is the one that matters: Akash's standing rule is that nothing sent to an API
becomes training data, and most providers' terms prohibit using their outputs to train
another model - a student trained on them would sit in production legally tainted. Running
weights we are licensed to run has neither problem, and we can regenerate the set whenever
we like without anyone's permission.

Both halves of a pair come from the teacher. The MESSAGE is training data just as much as
the card, so it cannot come from a different source than the targets.

Two calls per pair, not one. Asking for a message and its summary together lets the model
write a message to fit a summary it has already planned, which teaches the student a
relationship that will not exist at inference time. So the teacher writes a message, then
summarises it in a separate call with no memory of why it wrote it.

    # dry run on CPU with a tiny model, to prove the cells work
    python data_gen/distill_pairs.py --n 4 --model Qwen/Qwen2.5-0.5B-Instruct --out /tmp/x.jsonl
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Variation is specified, not hoped for. 2,000 samples drawn from one prompt are 2,000
# near-copies; the research on this is consistent that a few hundred DIVERSE filtered
# examples beat tens of thousands of raw ones.
TOPICS = [
    "a build or release going out", "a bug someone found while testing",
    "QA findings with counts and owners", "design review feedback on a mockup",
    "a deadline chase before a demo", "a post-mortem after an outage",
    "someone handing over work before leave", "a vendor or billing question",
    "app store or play store review feedback", "a security patch that has to ship",
    "a meeting being moved and what it blocks", "onboarding a new teammate",
    "a customer complaint relayed to the team", "infrastructure cost or capacity",
    "a feature being cut or descoped", "test devices and environments",
    "translation or localisation issues", "an API contract change for the apps",
    "someone blocked waiting on another team", "a config or permissions problem",
]
SHAPES = [
    "one long paragraph, no line breaks",
    "a numbered list of three or four items with a sentence of context first",
    "two short paragraphs separated by a blank line",
    "a rambling stream of thought that buries one real decision",
    "a status update that opens with pleasantries and ends with a sign-off",
    "a reply that quotes something someone else said and reacts to it",
    "mostly a list of links with labels and one real ask",
    "a short sharp message with two facts and no filler",
]
VOICES = [
    "a hurried engineer typing on mobile, lowercase, few full stops",
    "a careful manager who writes in complete sentences",
    "someone frustrated and venting a little before getting to the point",
    "a QA tester listing what they found, terse and numbered",
    "a founder writing late at night, long and a bit disorganised",
    "someone polite and formal, lots of please and thanks",
]
# A fifth of the set is pure chatter, so the student learns that the right answer is
# sometimes no card at all. Without these it will summarise a message about lunch.
CHATTER_SHARE = 0.2

WRITE_MESSAGE = """Write ONE realistic message from a software team's group chat.

Topic: {topic}
Shape: {shape}
Voice: {voice}
Length: {length} sentences.

{extra}

This is a message typed into a MESSAGING APP, not an email. No subject line, no "Dear",
no signature, no "Best regards". People write to each other directly.

Write ONLY the message text. No preamble, no quotes around it, no explanation.
English only. Use invented names and invented numbers."""

WRITE_CHATTER = """Write ONE realistic message from a software team's group chat that
contains NOTHING anyone would need to remember later - pure small talk. Weather, lunch,
weekend plans, a joke, a complaint about the coffee machine, idle chat about a TV show.

Shape: {shape}
Voice: {voice}
Length: {length} sentences.

It must NOT contain any deadline, instruction, decision, bug, owner or amount.
Write ONLY the message text. English only."""

SUMMARISE = """Summarize one chat message for the person who received it, for a "Summary"
tab they scroll back through later.

Keep ONLY what they must remember or do: deadlines, who owns what, amounts and counts,
decisions taken, anything broken or blocking, explicit instructions. Drop greetings,
opinions, hedging, thanks and small talk.

Each point is a complete phrase of three to fifteen words. Never a bare list item or a
single word. Rewrite in your own words - do not copy a sentence from the message whole.
Use as many points as the message needs: two if it says two things, eight if it says
eight. Never invent anything that is not in the message.

Quoted text inside the message is usually an EXAMPLE the sender is talking about, not an
instruction to the reader. Do not turn it into a point.

If the message has nothing worth remembering, return {{"title": null, "points": []}}.

Reply with JSON only: {{"title": "<two or three words, Title Case>", "points": [...]}}

The message:

{message}"""


def _json(reply):
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# The teacher drifts into writing email when asked for a "message": the dry run produced
# one opening "Subject: Infrastructure Costs and Capacity in Our Team" (2026-10-09). The
# prompt says not to; this catches the ones that do it anyway.
_EMAIL_SHAPED = re.compile(r"^\s*(?:subject|re|fwd)\s*:|^\s*dear\s+\w|"
                           r"\b(?:best regards|kind regards|sincerely|yours truly)\b",
                           re.IGNORECASE)


def validate(message, card):
    """Reject a pair rather than teach the student something wrong.

    Filtering is the part that matters: a few hundred clean pairs beat tens of thousands
    of raw ones, and a student will faithfully learn any mistake left in.
    """
    if not message or len(message.split()) < 15:
        return "message too short to need a summary"
    if re.search(r"[ऀ-ॿ一-鿿]", message):
        return "non-English script"
    if _EMAIL_SHAPED.search(message):
        return "written as an email, not a chat message"
    if card is None:
        return "teacher returned unparseable JSON"
    title, points = card.get("title"), card.get("points")
    if not isinstance(points, list):
        return "points is not a list"
    if title is None or not points:
        return None if not points and title is None else "half-empty card"
    if not isinstance(title, str) or not 1 <= len(title.split()) <= 5:
        return f"implausible title {title!r}"
    for p in points:
        if not isinstance(p, str):
            return "non-string point"
        words = p.split()
        if not 3 <= len(words) <= 20:
            return f"point of {len(words)} words"
        # The one thing the student must not learn is to copy sentences whole - that is
        # the failure the rules already have.
        for sentence in re.split(r"(?<=[.!?])\s+", message):
            s = sentence.strip(" .!?")
            if len(s) > 25 and s.lower() == p.strip(" .!?").lower():
                return "point is a verbatim sentence"
    return None


def build_specs(n, seed=0):
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        chatter = rng.random() < CHATTER_SHARE
        specs.append({
            "chatter": chatter,
            "topic": rng.choice(TOPICS),
            "shape": rng.choice(SHAPES),
            "voice": rng.choice(VOICES),
            "length": rng.choice([3, 4, 5, 6, 7, 8, 10, 12]),
            "extra": rng.choice([
                "Include a specific date or day and a specific number.",
                "Name at least one person who owns something.",
                "Include an amount of money or a count of items.",
                "Bury the important part in the middle, with filler either side.",
                "Mention something that is broken or blocking.",
                "Include one quoted example of what someone typed.",
                "", "",
            ]),
        })
    return specs


def generate(ask, n, seed=0, log=print):
    """`ask(prompt, max_tokens) -> str` is the teacher. Returns a list of pairs."""
    pairs, dropped = [], {}
    for i, spec in enumerate(build_specs(n, seed)):
        tmpl = WRITE_CHATTER if spec["chatter"] else WRITE_MESSAGE
        message = (ask(tmpl.format(**spec), 420) or "").strip().strip('"')
        message = re.sub(r"^(?:here'?s?|sure[,!]?|message:)\s*", "", message, flags=re.I).strip()
        card = _json(ask(SUMMARISE.format(message=message), 420))
        why = validate(message, card)
        if why:
            dropped[why] = dropped.get(why, 0) + 1
        else:
            pairs.append({"message": message, "card": card, "chatter": spec["chatter"],
                          "topic": spec["topic"]})
        if (i + 1) % 25 == 0:
            log(f"    {i+1}/{n} asked, {len(pairs)} kept")
    return pairs, dropped


def local_teacher(model_id, device="cpu"):
    """A transformers-backed `ask` for the dry run and for Colab alike."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    if device != "cpu":
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()

    def ask(prompt, max_tokens=420):
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=True,
                                 temperature=0.9, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/distill/pairs.jsonl")
    a = ap.parse_args()

    ask = local_teacher(a.model, a.device)
    pairs, dropped = generate(ask, a.n, a.seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    blank = sum(1 for p in pairs if not p["card"]["points"])
    print(f"\n  {len(pairs)}/{a.n} pairs kept -> {a.out}")
    print(f"  {blank} are chatter with an empty card "
          f"({100*blank/max(len(pairs),1):.0f}%)")
    if dropped:
        print("  dropped:")
        for why, k in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"    {k:>5}  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
