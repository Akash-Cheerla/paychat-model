"""Build colab_distill/train_student.ipynb - fine-tune the small model that ships.

The student learns `message -> card` from the teacher's 1,796 pairs. Two candidates, same
data, same split, same loop, so the difference in the score is the model:

  DistilBART-SAMSum  306M, already tuned on messenger dialogue, writes well out of the box
                     but loses content (49.2% recall as a stock checkpoint)
  flan-t5-base       250M, instruction-tuned, better at structured output

Three decisions worth knowing before reading the code:

TARGET FORMAT is flat text, not JSON. A 300M model spends real capacity on brackets and
quotes, and one malformed character loses the whole card. "Title | point | point" parses
just as reliably and leaves the capacity for the summary. JSON stays at the API boundary.

EMPTY CARDS are the literal token NONE. 20% of the set is chatter whose correct answer is
no card, and the student has to be able to say so - a model that always writes a card is
the iOS failure Akash rejected on sight.

A MANUAL LOOP rather than Seq2SeqTrainer. transformers 5.x renamed several Trainer
arguments; a hand-written loop is 25 lines, has no version surface, and every line of it
was executed on CPU before this notebook was handed over.

    python data_gen/make_train_student_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab_distill" / "train_student.ipynb"

# Shared by the notebook and the local dry run, so what is tested is what runs.
FORMAT = '''SEP = " | "
NONE = "NONE"

def card_to_target(card):
    """A card -> one line of text. Flat beats JSON for a 300M model."""
    pts = card.get("points") or []
    if not pts:
        return NONE
    return SEP.join([card.get("title") or "Notes"] + list(pts))

def target_to_card(text):
    """...and back. Anything malformed becomes "no card", never a crash."""
    text = (text or "").strip()
    if not text or text.upper().startswith(NONE):
        return {"title": None, "points": []}
    bits = [b.strip() for b in text.split(SEP.strip()) if b.strip()]
    if len(bits) < 2:
        return {"title": None, "points": []}
    return {"title": bits[0], "points": bits[1:]}
'''


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.splitlines(keepends=True)}


CELLS = [
    md("""# Train the student

Fine-tune a small model on the teacher's 1,796 pairs. This is the model that actually
ships — on the 8 GB box, ~400 MB, no GPU.

Two candidates trained identically so the difference is the model, not the setup:
**DistilBART-SAMSum** (306M, already tuned on messenger chat) and **flan-t5-base** (250M,
instruction-tuned, better at structured output).

Both are scored on a held-out split **and** on the five messages from the iOS screenshots —
which is the bar Akash actually set.

Everything is written to Drive, so a disconnect costs you the current epoch, not the run."""),

    md("## 1. Environment"),
    code("""!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip -q install -U "transformers>=4.44" accelerate sentencepiece
import torch, json, os, time, random, glob
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())"""),

    md("## 2. Drive and the pairs"),
    code('''from google.colab import drive
drive.mount("/content/drive")
WORK = "/content/drive/MyDrive/paychat_distill"
os.makedirs(WORK, exist_ok=True)

import shutil
if not glob.glob(f"{WORK}/pairs.jsonl"):
    if not glob.glob("/content/pairs.jsonl"):
        from google.colab import files
        print("upload the CLEANED pairs.jsonl:")
        files.upload()
    for c in glob.glob("/content/pairs.jsonl"):
        shutil.copy(c, f"{WORK}/pairs.jsonl")

rows = [json.loads(l) for l in open(f"{WORK}/pairs.jsonl", encoding="utf-8") if l.strip()]
empty = sum(1 for r in rows if not r["card"].get("points"))
print(f"{len(rows)} pairs | {empty} empty ({100*empty/len(rows):.0f}%)")
assert empty / len(rows) > 0.1, "expected ~20% empty cards — is this the CLEANED file?"
assert not any(r["message"].lower().startswith("here") for r in rows[:200]), \\
    "preambles present — run data_gen/clean_distill_pairs.py first"
print("looks like the cleaned set")'''),

    md("""## 3. Target format

Flat text, not JSON. A 300M model spends capacity on brackets and quotes, and one bad
character loses the card. `NONE` is how the student says "nothing worth summarising" —
20% of the training set, and the thing that stops it producing a card for small talk."""),
    code(FORMAT + '''
# self-test, so a format bug is caught here and not after an hour of training
_c = {"title": "Build 52", "points": ["Assigned in live", "Test the whole app"]}
assert target_to_card(card_to_target(_c)) == _c
assert target_to_card(card_to_target({"title": None, "points": []}))["points"] == []
assert target_to_card("garbage with no separator")["points"] == []
assert target_to_card("")["points"] == []
print("round-trip ok:", card_to_target(_c))'''),

    md("## 4. Split"),
    code('''random.Random(0).shuffle(rows)
n_val = max(60, len(rows) // 10)
val, train = rows[:n_val], rows[n_val:]
print(f"train {len(train)} | val {len(val)}")
print(f"val empty-card share: {100*sum(1 for r in val if not r['card']['points'])/len(val):.0f}%")'''),

    md("""## 5. Train

Manual loop on purpose — transformers 5.x renamed several `Seq2SeqTrainer` arguments, and
25 lines with no version surface beats debugging a rename on Colab."""),
    code('''from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader

MAX_IN, MAX_OUT = 512, 256
EPOCHS, BATCH, LR = 4, 4, 5e-5

def make_loader(tok, data, prefix, shuffle):
    def collate(batch):
        x = tok([prefix + b["message"] for b in batch], max_length=MAX_IN,
                truncation=True, padding=True, return_tensors="pt")
        y = tok([card_to_target(b["card"]) for b in batch], max_length=MAX_OUT,
                truncation=True, padding=True, return_tensors="pt")
        labels = y["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        x["labels"] = labels
        return x
    return DataLoader(data, batch_size=BATCH, shuffle=shuffle, collate_fn=collate)

def train_one(model_id, prefix=""):
    print(f"\\n{'='*70}\\ntraining {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loader = make_loader(tok, train, prefix, True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * len(loader), pct_start=0.1)

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for i, batch in enumerate(loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
            if i % 50 == 0:
                print(f"  ep{ep+1} {i}/{len(loader)}  loss {tot/(i+1):.3f}  "
                      f"{(time.time()-t0)/60:.0f}m")
        print(f"  epoch {ep+1} mean loss {tot/len(loader):.3f}")
    return tok, model

def predict(tok, model, messages, prefix="", bs=8):
    model.eval(); out = []
    for i in range(0, len(messages), bs):
        chunk = messages[i:i+bs]
        x = tok([prefix + m for m in chunk], max_length=MAX_IN, truncation=True,
                padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**x, max_length=MAX_OUT, num_beams=4, early_stopping=True)
        out += [target_to_card(tok.decode(o, skip_special_tokens=True)) for o in g]
    return out'''),

    md("""## 6. Scoring

`kept` uses word overlap, which **under-reports a model that paraphrases well** — it
scored the teacher at 51.6% when the examples showed it capturing everything. Treat it as
a floor and compare engines on it, never as an absolute. `empty_right` and the screenshot
cards are the honest signals."""),
    code('''import re, difflib
STOP = set("""a an the this that these those it its we you they them our your their my me i
in on at to for of with from by as into onto over under after before once when while
until during about across through between among per via and or but not no nor if then
is are was were be been being has have had do does did will would shall should can could
may might must need needs so because there here just also still only very really please
kindly make sure""".split())
cw = lambda t: [w for w in re.findall(r"[a-z0-9][a-z0-9'\\-]*", (t or "").lower())
                if w not in STOP and len(w) > 1]

def score(preds, gold):
    kept = miss = 0; eok = etot = 0
    ratios, copies, npts = [], 0, 0
    for p, g in zip(preds, gold):
        gp, pp = g["card"].get("points") or [], p["points"]
        if not gp:
            etot += 1; eok += (not pp); continue
        for want in gp:
            w = cw(want)
            best = max([sum(1 for x in w if x in set(cw(q)))/max(len(w),1)
                        for q in pp] or [0])
            kept += best >= 0.6; miss += best < 0.6
        sents = [s.strip(" .!?") for s in re.split(r"(?<=[.!?])\\s+", g["message"]) if s.strip()]
        for q in pp:
            npts += 1
            b, src = max(((difflib.SequenceMatcher(None, q.lower(), s.lower()).ratio(), s)
                          for s in sents), default=(0, ""))
            if b >= 0.97: copies += 1
            if src: ratios.append(len(q)/max(len(src),1))
    return {"kept": 100*kept/max(kept+miss,1),
            "empty_right": 100*eok/max(etot,1),
            "verbatim": 100*copies/max(npts,1),
            "length": 100*sum(ratios)/max(len(ratios),1),
            "pts": npts/max(sum(1 for p in preds if p["points"]),1)}

SCREENSHOTS = {
 "Build 52": "Build 52 assigned in live. Please test the entirety of the app. It is marked "
   "as a stable release. Ensure 0 issues in positive flow before we hand it over.",
 "Overlap Issues": "End call when top banner disappears. White space remains on all "
   "modules. Ideal spacing should revert back to original once banner disappears. This is "
   "creating overlap issues on certain modules.",
 "Voice Note and Call Issues": "The voice note saves in 15-20 seconds which is far too "
   "slow for a normal message. It also auto-disconnects calls after 2-3 attempts, every "
   "single time.",
 "Verify Build": "Verify Ride Intents in the new build. Verify Locations under Settings as "
   "well. Verify the fixed bugs from last week before we sign this off.",
 "IOS Issue  (must produce NOTHING)": "IOS. Android. Ride Request. Reply. IOS Issue. IOS. "
   "Android. Ride Request. Reply. IOS Issue. IOS Android Ride Request Reply.",
}'''),

    md("## 7. Run both"),
    code('''CANDIDATES = [("philschmid/distilbart-cnn-12-6-samsum", ""),
              ("google/flan-t5-base", "summarize the chat message: ")]
results = {}

for model_id, prefix in CANDIDATES:
    tok, model = train_one(model_id, prefix)
    preds = predict(tok, model, [r["message"] for r in val], prefix)
    s = score(preds, val)
    results[model_id] = s
    print(f"\\n  {model_id}")
    print(f"    kept {s['kept']:.1f}% | empty right {s['empty_right']:.0f}% | "
          f"verbatim {s['verbatim']:.1f}% | length {s['length']:.0f}% | {s['pts']:.1f} pts/card")

    print("\\n    --- the screenshot messages ---")
    shots = predict(tok, model, list(SCREENSHOTS.values()), prefix)
    for name, card in zip(SCREENSHOTS, shots):
        print(f"\\n    iOS: {name}")
        print(f"    student: {card['title']}")
        for p in card["points"]:
            print(f"       - {p}")

    out = f"{WORK}/student_{model_id.split('/')[-1]}"
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"\\n    saved {out}")
    del model; torch.cuda.empty_cache()'''),

    md("## 8. Compare"),
    code('''print(f"{'model':<40}{'kept':>8}{'empty':>8}{'verbat':>8}{'len':>7}{'pts':>6}")
for k, s in results.items():
    print(f"{k.split('/')[-1]:<40}{s['kept']:>7.1f}%{s['empty_right']:>7.0f}%"
          f"{s['verbatim']:>7.1f}%{s['length']:>6.0f}%{s['pts']:>6.1f}")
print("\\nreference — rules 76.7% kept / 52.3% verbatim | teacher ~7 pts/card")
print("the screenshot cards above matter more than any of these numbers")'''),

    md("""---

**Then:** zip the winner from Drive, and it runs locally through `summary_engines.py` so
it can be scored on the dogfood log the same way everything else was."""),
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
