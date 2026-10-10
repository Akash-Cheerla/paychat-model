"""Build colab_distill/resummarise.ipynb - re-write the CARDS for messages we already have.

The first run produced 1,884 good messages with bad cards: the teacher scored 31% recall
against our labelled set because the summarise prompt pushed brevity so hard it compressed
by dropping. The messages cost most of the 6.7 hours and are fine - 100% unique, all 20
topics, 20% chatter as designed, no hallucinated names. Only the cards need redoing, which
is ONE teacher call per pair instead of two.

Two things this notebook does that the first one did not, both because that run was lost
to a disconnect after six hours:

  Drive, not /content. Colab wipes its local filesystem when a session dies, which is why
  out/ was empty afterwards. Partials go to Drive so a disconnect costs minutes.

  Batched generation. The first run called the model once per message; a GPU will do eight
  at a time for roughly the same wall clock. Expect ~45-60 min instead of ~3 hours.

Self-contained on purpose - it does not depend on anything defined in the first notebook,
because by the time you need it that session is usually gone.

    python data_gen/make_resummarise_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab_distill" / "resummarise.ipynb"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.splitlines(keepends=True)}


CELLS = [
    md("""# Re-summarise — fix the cards, keep the messages

The first run made 1,884 good messages with bad cards. The teacher scored **31% recall**
against our labelled set: the prompt pushed brevity so hard it compressed by *dropping*
content, producing 3.2 points per card where the reference teacher produced 7.0.

The messages are fine — 100% unique, all 20 topics, 20% chatter as designed, no invented
names. So this only rewrites the cards: **one teacher call per pair instead of two.**

Two fixes over the first notebook, both learned the hard way:

* **Partials go to Google Drive.** Colab wipes `/content` when a session dies — that's why
  `out/` was empty after the last disconnect. On Drive, a disconnect costs minutes.
* **Batched generation.** Eight messages per forward pass instead of one. ~45–60 min
  rather than ~3 hours.

Upload `distill_pairs_1884.jsonl` when cell 3 asks for it."""),

    md("## 1. Environment"),
    code("""!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip -q install -U "transformers>=4.44" accelerate bitsandbytes sentencepiece
import torch, json, time, os, glob
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())"""),

    md("""## 2. Drive

Everything is written here, so a disconnect does not cost you the run. Re-running this
notebook after a drop picks up from the last save."""),
    code('''from google.colab import drive
drive.mount("/content/drive")
WORK = "/content/drive/MyDrive/paychat_distill"
os.makedirs(WORK, exist_ok=True)
PART = f"{WORK}/resummarised_partial.jsonl"
print("working dir:", WORK)
print("existing:", os.listdir(WORK))'''),

    md("## 3. The messages"),
    code('''import shutil

if not glob.glob(f"{WORK}/distill_pairs_*.jsonl"):
    if not glob.glob("/content/distill_pairs_*.jsonl"):
        from google.colab import files
        print("upload distill_pairs_1884.jsonl:")
        files.upload()
    # copy, not os.replace: Drive is a FUSE mount and /content is local disk, so a
    # rename across them raises "Invalid cross-device link".
    for c in glob.glob("/content/distill_pairs_*.jsonl"):
        shutil.copy(c, f"{WORK}/{os.path.basename(c)}")

SRC = sorted(glob.glob(f"{WORK}/distill_pairs_*.jsonl"))[-1]
pairs = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
print(f"{len(pairs)} pairs from {SRC}")
print("sample:", " ".join(pairs[0]["message"].split())[:140])'''),

    md("""## 4. The teacher

14B in 4-bit. `padding_side="left"` matters — a decoder-only model pads on the left or the
batched outputs come back misaligned with their prompts."""),
    code('''from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TEACHER = "Qwen/Qwen2.5-14B-Instruct"
BATCH   = 8

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(TEACHER)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(TEACHER, quantization_config=bnb,
                                             device_map="auto")
model.eval()

def ask_batch(prompts, max_tokens=500):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    n_in = enc["input_ids"].shape[1]
    return [tok.decode(o[n_in:], skip_special_tokens=True) for o in out]

print(ask_batch(["Reply with the single word: ready"])[0][:40])'''),

    md("""## 5. The prompt and the validator

The fixed prompt is **in this cell**, not a separate one you might forget to run. The
earlier version said "as many points as the message needs", which a 14B read as licence to
be terse. This one says omitting is worse than over-writing, and that single change took
points per card from 3.2 to 6.9."""),
    code('''import re

SUMMARISE = """Summarize one chat message for the person who received it, for a "Summary"
tab they scroll back through later.

Write one point for EVERY distinct thing worth remembering: every deadline, every person
who owns something, every amount or count, every decision taken, everything broken or
blocking, every explicit instruction. Do not merge two facts into one point. Do not stop
early. Omitting something is far worse than writing one point too many - most real
messages need four to eight points.

Drop only greetings, opinions, hedging, thanks and small talk.

Each point is a complete phrase of three to fifteen words, in your own words - do not copy
a sentence from the message whole. Never invent anything that is not in the message.

Quoted text inside the message is usually an EXAMPLE the sender is talking about, not an
instruction to the reader. Do not turn it into a point.

If the message has nothing worth remembering, return {{"title": null, "points": []}}.

Reply with JSON only: {{"title": "<two or three words, Title Case>", "points": [...]}}

The message:

{message}"""

_EMAIL_SHAPED = re.compile(r"^\\s*(?:subject|re|fwd)\\s*:|^\\s*dear\\s+\\w|"
                           r"\\b(?:best regards|kind regards|sincerely)\\b", re.IGNORECASE)

def _json_of(reply):
    m = re.search(r"\\{.*\\}", reply or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def validate(message, card):
    if not message or len(message.split()) < 15:
        return "message too short"
    if _EMAIL_SHAPED.search(message):
        return "written as an email"
    if card is None:
        return "unparseable JSON"
    title, points = card.get("title"), card.get("points")
    if not isinstance(points, list):
        return "points not a list"
    if title is None or not points:
        return None if not points and title is None else "half-empty card"
    if not isinstance(title, str) or not 1 <= len(title.split()) <= 5:
        return f"implausible title {title!r}"
    for p in points:
        if not isinstance(p, str):
            return "non-string point"
        if not 3 <= len(p.split()) <= 20:
            return f"point of {len(p.split())} words"
        for s in re.split(r"(?<=[.!?])\\s+", message):
            s = s.strip(" .!?")
            if len(s) > 25 and s.lower() == p.strip(" .!?").lower():
                return "verbatim sentence"
    return None

print("ready — prompt asks for 4-8 points")'''),

    md("""## 6. Run

Saves to Drive every batch. If the session drops, re-run this cell and it resumes."""),
    code('''done = []
if os.path.exists(PART):
    done = [json.loads(l) for l in open(PART, encoding="utf-8") if l.strip()]
    print(f"resuming — {len(done)} already written")
seen = {d["message"] for d in done}
todo = [p for p in pairs if p["message"] not in seen]
print(f"{len(todo)} to do")

dropped, t0 = {}, time.time()
with open(PART, "a", encoding="utf-8") as sink:
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        replies = ask_batch([SUMMARISE.format(message=p["message"]) for p in chunk])
        for p, r in zip(chunk, replies):
            card = _json_of(r)
            why = validate(p["message"], card)
            if why:
                dropped[why] = dropped.get(why, 0) + 1
            else:
                rec = {**p, "card": card}
                done.append(rec)
                sink.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        sink.flush()
        if (i // BATCH) % 5 == 0:
            mins = (time.time() - t0) / 60
            per = mins / max(i + len(chunk), 1)
            print(f"  {i+len(chunk)}/{len(todo)}  kept {len(done)}  "
                  f"{mins:.0f}m elapsed, ~{per*(len(todo)-i-len(chunk)):.0f}m left")

print(f"\\n{len(done)} cards in {(time.time()-t0)/60:.0f} min")
for why, k in sorted(dropped.items(), key=lambda kv: -kv[1]):
    print(f"  {k:>5}  {why}")'''),

    md("""## 7. Is it good enough to train on?

`points/card` is the one that proves the prompt fix held — it was 3.2 before, 6.9 on the
labelled set after. `verbatim` must stay near zero: if the teacher copies sentences, the
student learns to copy sentences and we have rebuilt the rules."""),
    code('''import difflib
carded = [p for p in done if p["card"]["points"]]
blank = len(done) - len(carded)
npts = sum(len(p["card"]["points"]) for p in carded)

ratios, copies = [], 0
for p in carded:
    sents = [s.strip(" .!?") for s in re.split(r"(?<=[.!?])\\s+", p["message"]) if s.strip()]
    for x in p["card"]["points"]:
        best, src = max(((difflib.SequenceMatcher(None, x.lower(), s.lower()).ratio(), s)
                         for s in sents), default=(0, ""))
        if best >= 0.97:
            copies += 1
        if src:
            ratios.append(len(x) / max(len(src), 1))

print(f"{len(done)} pairs, {npts} points")
print(f"  points/card : {npts/max(len(carded),1):.1f}   (was 3.2, target ~7)")
print(f"  empty cards : {blank} ({100*blank/max(len(done),1):.0f}%)")
print(f"  verbatim    : {100*copies/max(npts,1):.1f}%   (want < 5%)")
print(f"  length      : {100*sum(ratios)/max(len(ratios),1):.0f}%   (want < 75%)")'''),

    md("## 8. Ten samples — read these before trusting the numbers"),
    code('''import random
for p in random.Random(0).sample(carded, min(10, len(carded))):
    print("=" * 76)
    print("MSG :", " ".join(p["message"].split())[:260])
    print("CARD:", p["card"]["title"])
    for x in p["card"]["points"]:
        print("   -", x)'''),

    md("## 9. Save"),
    code('''FINAL = f"{WORK}/distill_pairs_fixed_{len(done)}.jsonl"
with open(FINAL, "w", encoding="utf-8") as f:
    for p in done:
        f.write(json.dumps(p, ensure_ascii=False) + "\\n")
print("on Drive:", FINAL, f"{os.path.getsize(FINAL)/1e6:.1f} MB")

from google.colab import files
files.download(FINAL)'''),

    md("""---

It is on Drive either way, so if the download fails you have not lost anything. Put it in
`data/distill/` locally and the next step is training the student."""),
]


def main():
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"  {OUT.relative_to(ROOT)}  "
          f"({len(CELLS)} cells, {sum(1 for c in CELLS if c['cell_type']=='code')} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
