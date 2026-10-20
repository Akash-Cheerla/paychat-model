"""Build colab_distill/recap_round.ipynb - the training round for the conversation recap.

The rules already pick the right lines out of a window and compute the money. What they
cannot do is WRITE: five attempts at grouping and titling with rules produced "Too",
"Saturday Kitchen" and "I'll Plumber", and the shipped per-message student handed the same
lines wrote "owe 680 from dinner" when 680 was the cab fare.

Akash, 2026-10-19: "cant keep 1-2 lines and for each small topic and just 1-3 points we
can have sumary for each, thats not professional" - the selected-lines version is a record
of the conversation rather than a recap of it. It lists the steps where a person writes the
conclusion.

So the student learns ONE job: selected lines -> a written recap.

    rules   decide which lines matter, and compute the per-person share
    teacher writes the recap from those lines
    student learns the writing, and runs on Samyak's 4 vCPU box

WHAT THE STUDENT NEVER HAS TO DO IS ARITHMETIC. "18000 split three ways" is 6000 each, and
this is the model class that wrote "owe 680 from dinner". app.py computes shares already
and scores 194/195 on tests/test_split_matrix.py, so the figure arrives in the INPUT marked
[computed] and the student only places it in a sentence. Any number it emits that is not in
its input is refused at inference - the same grounding guard the headings use.

INPUT IS THE SELECTED LINES, NOT THE WINDOW. A busy room-day is 1,770-2,467 tokens against
flan-t5's 512, which is what made a window-level model look impossible. Twelve selected
lines are about 150 tokens, so no new base model is needed.

TEACHER IS OPEN WEIGHT ON OUR OWN HARDWARE. Akash's rule is that nothing sent to an API
becomes training data, and separately most providers forbid training another model on their
output - a student trained that way sits in production legally tainted.

    python data_gen/make_recap_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab_distill" / "recap_round.ipynb"

PAIRS = "/content/drive/MyDrive/paychat/recap_pairs.jsonl"
MODEL_OUT = "/content/drive/MyDrive/paychat/recap_model"


def md(t):
    return {"cell_type": "markdown", "metadata": {},
            "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}


CELLS = [
md("""# Conversation recap - training round

The student learns **selected lines -> written recap**. Nothing else.

| | |
|---|---|
| rules | pick the lines, compute the share |
| teacher | Qwen2.5-14B-Instruct, 4-bit, on this GPU |
| student | flan-t5-base, 250M, runs on the 4 vCPU box |

Run the cells in order. Phase 1 writes pairs to Drive and can be stopped and resumed -
it appends. Phase 3 trains. Nothing here calls an API.
"""),

code("""# Phase 0 - environment
!pip -q install -U "transformers>=4.44" accelerate bitsandbytes sentencepiece datasets
import torch, json, os, random, re, sys
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
from google.colab import drive
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/paychat', exist_ok=True)"""),

code("""# Phase 0b - the repo, for the rules that do the selecting and the arithmetic
# The BRANCH matters: distill_recaps.py and the reworked room_summarizer.py live on
# recap-round, not on main, because main is what Samyak deploys and this has not been
# approved for it. A clone of main fails with "No module named 'distill_recaps'".
BRANCH = "recap-round"
!test -d /content/paychat-model || git clone -q --branch $BRANCH https://github.com/Akash-Cheerla/paychat-model.git /content/paychat-model
%cd /content/paychat-model
!git fetch -q origin $BRANCH && git checkout -q $BRANCH && git pull -q origin $BRANCH
sys.path.insert(0, '/content/paychat-model')
sys.path.insert(0, '/content/paychat-model/data_gen')
try:
    import room_summarizer as R, distill_recaps as DR
except ModuleNotFoundError as e:
    raise SystemExit(
        f"{e}. The checkout is missing a file this notebook needs - you are probably on "
        f"main. Run:  !cd /content/paychat-model && git checkout {BRANCH} && git pull") from e
print("rules loaded - MIN_POINTS", R.MIN_POINTS, "| MAX_POINTS", R.MAX_POINTS)
print("branch:", open('/content/paychat-model/.git/HEAD').read().strip())"""),

md("""## Phase 1 - the teacher writes the pairs

Two calls per pair, never one. Asking for a conversation and its recap together lets the
model write a conversation to fit a recap it has already planned, which teaches the
student a relationship that will not exist at inference time.

The window is training data as much as the recap, so both halves come from the teacher.
Nothing from the dogfood logs or the WhatsApp exports is used - those are measurement
only, and real people's messages are never training data."""),

code("""# Phase 1 - load the teacher in 4-bit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
TEACHER = "Qwen/Qwen2.5-14B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
t_tok = AutoTokenizer.from_pretrained(TEACHER)
t_model = AutoModelForCausalLM.from_pretrained(TEACHER, quantization_config=bnb,
                                               device_map="auto")
t_model.eval()
print("teacher ready")"""),

code("""# Phase 1 - generate. Resumable: it appends, so re-run after a disconnect.
PAIRS = "%s"
# TWO STAGES, deliberately. Run 40 first and look at them: the whole round rests on the
# teacher writing CONCLUSIONS rather than restating the selected lines, and 40 pairs is
# fifteen minutes against six to nine hours for the full set. Five rules-based attempts at
# this feature looked right before being measured and were not, so the cheap check first.
# Raise this to 1800 once the 40 read well - the cell appends, so nothing is wasted.
TARGET_PAIRS = 40

def ask(prompt, max_new=800):
    msgs = [{"role": "user", "content": prompt}]
    text = t_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = t_tok(text, return_tensors="pt", truncation=True, max_length=3072).to(t_model.device)
    with torch.no_grad():
        out = t_model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                               temperature=0.9, top_p=0.95,
                               pad_token_id=t_tok.eos_token_id)
    return t_tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

have = sum(1 for _ in open(PAIRS)) if os.path.exists(PAIRS) else 0
print("already have", have)
rnd = random.Random(1234 + have)
skipped = {}
with open(PAIRS, "a") as f:
    while have < TARGET_PAIRS:
        setting = rnd.choice(DR.SETTINGS)
        topics = rnd.choice(DR.TOPIC_COUNTS)
        n = rnd.choice([22, 26, 30, 34, 38])
        chat = ask(DR.WINDOW_PROMPT.format(setting=setting, topics=topics, n=n,
                                           example=DR.WINDOW_EXAMPLE), 900)
        msgs = DR.parse_chat(chat)
        if len(msgs) < R.MIN_MESSAGES:
            skipped["unparsed"] = skipped.get("unparsed", 0) + 1
            continue
        summ = R.summarize_room(R.window(msgs), room="distill")
        if not summ:
            skipped["nothing selected"] = skipped.get("nothing selected", 0) + 1
            continue
        lines = [f"- {p['text']}" for p in summ.points]
        share = DR.computed_share(summ.points)
        if share:
            lines.append(f"- [computed] {share}")
        recap = ask(DR.RECAP_PROMPT.format(points="\\n".join(lines)), 420)
        f.write(json.dumps({"input": "\\n".join(lines), "target": recap,
                            "setting": setting, "selected": [p["text"] for p in summ.points],
                            "computed": share}, ensure_ascii=False) + "\\n")
        f.flush()
        have += 1
        if have %% 25 == 0:
            print(have, "pairs |", skipped)
print("done:", have, "pairs |", skipped)""" % PAIRS),

md("""## Phase 1b - READ THE 40 BEFORE GOING FURTHER

Stop here on the first run. The question is whether the teacher writes the conclusion or
just reformats the lines it was given. Send these back before raising TARGET_PAIRS.

```python
for p in [json.loads(l) for l in open(PAIRS)][:6]:
    print("INPUT\n" + p["input"])
    print("\nTEACHER\n" + p["target"])
    print("=" * 70)
```

What good looks like: the four lines about a split collapse into one figure, the owner is
named, and nothing appears that nobody said. What failure looks like: the same bullets
with headings on top - that is what the rules already give, and no amount of training
data turns it into a recap."""),

md("""## Phase 2 - throw away the pairs that would teach the wrong thing

The teacher is good, not perfect. A pair is dropped when:

- the recap contains a **number that is not in its input** - that is the "owe 680 from
  dinner" failure, and training on it teaches exactly that
- the recap is empty, or is one of the section headings with nothing under it
- the recap restates the input line for line, which teaches copying rather than writing
- the recap is longer than its input, which is not a summary"""),

code("""# Phase 2 - clean
import re
NUM = re.compile(r"\\d[\\d,]*(?:\\.\\d+)?")
def norm(s): return re.sub(r"[,\\s]", "", s.lower())

raw = [json.loads(l) for l in open(PAIRS) if l.strip()]
kept, dropped = [], {}
for p in raw:
    inp, tgt = p["input"], (p["target"] or "").strip()
    if len(tgt) < 25:
        dropped["empty"] = dropped.get("empty", 0) + 1; continue
    src = norm(inp)
    invented = [t for t in NUM.findall(tgt) if norm(t) not in src]
    if invented:
        dropped["invented a number"] = dropped.get("invented a number", 0) + 1; continue
    if len(tgt) > len(inp) * 1.1:
        dropped["longer than its input"] = dropped.get("longer than its input", 0) + 1; continue
    in_lines = {l.strip("- ").strip().lower() for l in inp.splitlines() if l.strip()}
    tg_lines = [l.strip("- ").strip().lower() for l in tgt.splitlines()
                if l.strip().startswith("-")]
    if tg_lines and sum(1 for l in tg_lines if l in in_lines) / len(tg_lines) > 0.6:
        dropped["copied the input"] = dropped.get("copied the input", 0) + 1; continue
    if not re.search(r"(?:DECISIONS|ACTION ITEMS|OPEN QUESTIONS|INFORMATION)", tgt):
        dropped["no sections"] = dropped.get("no sections", 0) + 1; continue
    # A figure that is only ever ASKED about must not be asserted. The local validation
    # run produced a chat saying "so we're paying $9,600 each?" and a recap saying
    # "We're paying $9,600 each." - the grounding check passes it, because 9,600 is in
    # the input, and it is still a number nobody agreed to.
    asked_only = set()
    for line in inp.splitlines():
        nums = set(NUM.findall(line))
        if not nums: continue
        if line.rstrip().endswith("?"):
            asked_only |= nums
        else:
            asked_only -= nums
    if asked_only:
        head = tgt.split("OPEN QUESTIONS")[0]
        if any(n in head for n in asked_only):
            dropped["asserted a question"] = dropped.get("asserted a question", 0) + 1
            continue
    kept.append({"input": inp, "target": tgt})
print(f"{len(raw)} generated -> {len(kept)} usable")
print("dropped:", dropped)
json.dump(kept, open('/content/drive/MyDrive/paychat/recap_clean.json', 'w'))
print(kept[0]["input"][:300]); print("---"); print(kept[0]["target"][:300])"""),

md("""## Phase 3 - train the student

flan-t5-base, 250M. A hand-written loop rather than Seq2SeqTrainer: transformers renamed
several Trainer arguments between versions, and 25 lines with no version surface is
cheaper than debugging a notebook on someone else's GPU.

The held-out split is cut BEFORE training and never touched, because the question this
has to answer is whether the student writes recaps for windows it has not seen."""),

code("""# Phase 3 - train
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
import torch.nn.functional as F

data = json.load(open('/content/drive/MyDrive/paychat/recap_clean.json'))
random.Random(7).shuffle(data)
cut = max(60, len(data) // 10)
heldout, train = data[:cut], data[cut:]
print(len(train), "train |", len(heldout), "held out")

BASE = "google/flan-t5-base"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE).cuda()
PREFIX = "recap: "

def batches(rows, bs=8, shuffle=True):
    idx = list(range(len(rows)))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [rows[j] for j in idx[i:i + bs]]
        x = tok([PREFIX + r["input"] for r in chunk], return_tensors="pt",
                padding=True, truncation=True, max_length=512)
        y = tok([r["target"] for r in chunk], return_tensors="pt",
                padding=True, truncation=True, max_length=320)
        labels = y.input_ids.masked_fill(y.input_ids == tok.pad_token_id, -100)
        yield {k: v.cuda() for k, v in x.items()} | {"labels": labels.cuda()}

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
EPOCHS = 4
for ep in range(EPOCHS):
    model.train(); tot = n = 0
    for b in batches(train):
        loss = model(**b).loss
        loss.backward(); opt.step(); opt.zero_grad()
        tot += loss.item(); n += 1
    print(f"epoch {ep + 1}/{EPOCHS}  loss {tot / max(n,1):.4f}")
model.save_pretrained("%s"); tok.save_pretrained("%s")
print("saved")""" % (MODEL_OUT, MODEL_OUT)),

md("""## Phase 4 - the number that decides whether this ships

Not loss. Three things, on the held-out split the student has never seen:

| | why |
|---|---|
| **invented numbers** | has to be 0. A recap that moves an amount between facts is worse than no recap - it is the failure that killed the previous attempt |
| **conclusions, not steps** | the recap should be SHORTER than its input. If it is the same length it is a reformatted list, which is what we already have |
| **sections present** | the structure is the product |

A run that trains cleanly and invents numbers is a failed run."""),

code("""# Phase 4 - evaluate
model.eval()
def write(inp):
    x = tok(PREFIX + inp, return_tensors="pt", truncation=True, max_length=512).to('cuda')
    with torch.no_grad():
        out = model.generate(**x, max_length=320, num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)

invented = copied = nosec = 0
ratios = []
for r in heldout:
    got = write(r["input"])
    src = norm(r["input"])
    if [t for t in NUM.findall(got) if norm(t) not in src]:
        invented += 1
    ratios.append(len(got) / max(len(r["input"]), 1))
    if not re.search(r"(?:DECISIONS|ACTION ITEMS|OPEN QUESTIONS|INFORMATION)", got):
        nosec += 1
    in_lines = {l.strip("- ").strip().lower() for l in r["input"].splitlines() if l.strip()}
    tg = [l.strip("- ").strip().lower() for l in got.splitlines() if l.strip().startswith("-")]
    if tg and sum(1 for l in tg if l in in_lines) / len(tg) > 0.6:
        copied += 1

n = len(heldout)
print(f"held out: {n}")
print(f"  invented a number     {invented}/{n}  {invented/n:.1%}   MUST BE 0")
print(f"  no sections           {nosec}/{n}  {nosec/n:.1%}")
print(f"  copied the input      {copied}/{n}  {copied/n:.1%}")
print(f"  length vs input       {sum(ratios)/len(ratios):.0%}   (want well under 100%)")
print("\\n--- three held-out examples ---")
for r in heldout[:3]:
    print("\\nINPUT\\n" + r["input"][:400])
    print("\\nSTUDENT\\n" + write(r["input"])[:400])"""),

md("""## Phase 5 - bring it back

Download `recap_model` from Drive to `recap_model/` in the repo, then locally:

```bash
python tests/test_conversation_summary.py     # the rules half, unchanged
python tests/test_no_leaks.py                 # nothing leaks through the writing
```

The guard that refuses a number absent from the input lives in the serving code, not
here, so a bad generation degrades to the selected-lines recap rather than shipping a
wrong amount."""),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    code_cells = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"  {OUT.relative_to(ROOT)}  ({len(CELLS)} cells, {code_cells} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
