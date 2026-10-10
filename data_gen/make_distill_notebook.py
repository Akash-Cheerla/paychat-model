"""Build colab_distill/generate_pairs.ipynb - one session that writes the training set.

The teacher is an open-weight model on Colab's GPU, not an API. Akash's rule is that
nothing sent to an API becomes training data, and most providers' terms forbid using their
outputs to train another model - a student trained on them would sit in production legally
tainted. Weights we are licensed to run have neither problem, and the set can be
regenerated any time without asking anyone.

The generation logic lives in data_gen/distill_pairs.py and is INLINED into the notebook
rather than cloned, so the notebook is self-contained and what ran is visible in the
output. It is dry-run on CPU with a 0.5B before this builder is used - see the repo rule
about executing every cell locally first.

    python data_gen/make_distill_notebook.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab_distill" / "generate_pairs.ipynb"

LOGIC = (ROOT / "data_gen" / "distill_pairs.py").read_text(encoding="utf-8")
# Everything above main(): the prompts, the specs, generate() and validate().
LOGIC = LOGIC.split("def local_teacher(")[0]
# A notebook has no __file__, so the module-level ROOT line raises NameError on the very
# first cell that matters. Caught by dry-running the inlined cell locally (2026-10-09).
LOGIC = "\n".join(line for line in LOGIC.splitlines()
                  if not line.startswith("ROOT = Path(__file__)")
                  and line.strip() not in ("import sys", "from pathlib import Path",
                                           "import argparse"))


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


CELLS = [
    md("""# Summary distillation — generate the training set

A teacher writes 2,000 (chat message → summary card) pairs. A ~300M student is trained on
them afterwards and is what actually ships, on the 8 GB box.

**Why distillation.** Measured 2026-10-09 on our own data: the rules keep 76.7% of what
matters but 60.7% of their bullets are verbatim copies of a sentence — they select, they
do not summarise. Every small stock model rewrites properly (~5% verbatim) and keeps only
13–49%. A large model does both (80.7% kept, 1% verbatim) but will not fit in 8 GB. So the
student copies the large model instead of being one.

**Runtime:** pick a GPU. A T4 is enough for a 14B in 4-bit. Expect roughly 1.5–3 hours for
2,000 pairs (two generations per pair).

**Nothing real goes through here.** Every message is invented by the teacher. No dogfood,
no exports, no customer text."""),

    md("## 1. Environment"),
    code("""!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip -q install -U "transformers>=4.44" accelerate bitsandbytes sentencepiece
import torch, json, time
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())"""),

    md("""## 2. The teacher

14B in 4-bit fits a T4 with room to spare. Swap `TEACHER` for a larger model if you have
an A100 — the student only ever sees the output, so a better teacher is a better student,
up to the point where the student's capacity runs out."""),
    code('''TEACHER = "Qwen/Qwen2.5-14B-Instruct"   # fits a T4 in 4-bit
N_PAIRS = 2000
SEED    = 0

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(TEACHER)
model = AutoModelForCausalLM.from_pretrained(TEACHER, quantization_config=bnb,
                                             device_map="auto")
model.eval()

def ask(prompt, max_tokens=420):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=True,
                             temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

print(ask("Reply with the single word: ready"))'''),

    md("""## 3. The generation logic

Inlined from `data_gen/distill_pairs.py` so this notebook is self-contained and the
output shows exactly what ran.

Two calls per pair, deliberately. Asking for a message and its summary in one go lets the
model write a message to fit a summary it already planned, which teaches the student a
relationship that will not exist at inference time. So the teacher writes a message, then
summarises it in a separate call."""),
    code(LOGIC.rstrip()),

    md("""## 4. Generate

Filtering is the part that matters — a few hundred clean pairs beat tens of thousands of
raw ones, and a student faithfully learns any mistake left in. Watch the dropped counts:
if one reason dominates, fix the prompt and rerun rather than training on what got
through."""),
    code('''t0 = time.time()
pairs, dropped = generate(ask, N_PAIRS, SEED)
mins = (time.time() - t0) / 60

print(f"\\n{len(pairs)}/{N_PAIRS} kept in {mins:.0f} min")
blank = sum(1 for p in pairs if not p["card"]["points"])
print(f"{blank} chatter pairs with an empty card ({100*blank/max(len(pairs),1):.0f}%)")
for why, k in sorted(dropped.items(), key=lambda kv: -kv[1]):
    print(f"  {k:>5}  {why}")'''),

    md("## 5. Look at ten before trusting two thousand"),
    code('''import random
for p in random.Random(0).sample(pairs, min(10, len(pairs))):
    print("=" * 78)
    print("MSG :", " ".join(p["message"].split())[:300])
    print("CARD:", p["card"]["title"])
    for x in p["card"]["points"]:
        print("   -", x)'''),

    md("""## 6. Sanity numbers

The two that decide whether this set is worth training on. **Verbatim** must be near zero —
if the teacher copies sentences, the student learns to copy sentences and we have rebuilt
the rules. **Length** should be well under 100% of the source."""),
    code('''import difflib, re as _re

def sentences_of(t):
    return [s.strip(" .!?") for s in _re.split(r"(?<=[.!?])\\s+", t) if s.strip()]

ratios, copies, total = [], 0, 0
for p in pairs:
    sents = sentences_of(p["message"])
    for x in p["card"]["points"]:
        total += 1
        best, src = max(((difflib.SequenceMatcher(None, x.lower(), s.lower()).ratio(), s)
                         for s in sents), default=(0, ""))
        if best >= 0.97:
            copies += 1
        if src:
            ratios.append(len(x) / max(len(src), 1))

print(f"{total} points")
print(f"  verbatim copies : {100*copies/max(total,1):.1f}%   (want < 5%)")
print(f"  length vs source: {100*sum(ratios)/max(len(ratios),1):.0f}%   (want < 75%)")
print(f"  points per card : {total/max(sum(1 for p in pairs if p['card']['points']),1):.1f}")'''),

    md("## 7. Save and download"),
    code('''import os
os.makedirs("out", exist_ok=True)
path = f"out/distill_pairs_{len(pairs)}.jsonl"
with open(path, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\\n")
print(path, os.path.getsize(path) / 1e6, "MB")

from google.colab import files
files.download(path)'''),

    md("""---

**Next:** put the file in `data/distill/` and train the student. Candidates are
DistilBART-SAMSum (306M, already chat-tuned) and flan-t5-base (250M, better at structured
output) — same pairs, same eval, pick the winner on `tests/bakeoff_summary.py` rather than
on taste."""),
]


def main():
    nb = {"cells": CELLS, "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    code_cells = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"  {OUT.relative_to(ROOT)}  ({len(CELLS)} cells, {code_cells} code)")
    print(f"  inlined {len(LOGIC.splitlines())} lines of generation logic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
