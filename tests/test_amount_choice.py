"""Which of several figures in a thread is the amount?

The newest figure usually wins, and that is right for a correction - "dinner was $90" ...
"sorry it was $120 actually" should prompt for $120. It is wrong when the newest figure is
one somebody QUESTIONED and was told no:

    "The museum tickets were $18 per person. Can everyone send me that?"
    "Wait, I thought it was $15?"
    "No, that was the student rate. We paid full price."
        -> the sheet opened at $15

$15 is neither what was asked for nor what anyone agreed. Found 2026-10-08 in the
generated scenario set, documented as a known limitation when the split work shipped, and
fixed separately because it is amount SELECTION - the counter-offer path (FIRING_RULE 6e)
- rather than split arithmetic.

The fix needs both halves: the figure is doubted AND a later message refutes it. Doubt
alone would disqualify "can you send me $20?", which is a question and is also the ask.
The negative cases below matter more than the positive one.

    python tests/test_amount_choice.py
    python tests/test_amount_choice.py --url http://127.0.0.1:8971/detect
"""
import argparse
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app import _latest_amount          # noqa: E402


def h(*texts):
    return [{"text": t} for t in texts]


UNIT = [
    ("doubted then refuted: the doubted figure loses",
     h("The museum tickets were $18 per person. Can everyone send me that?",
       "Wait, I thought it was $15?",
       "No, that was the student rate. We paid full price."), "$18"),

    # Everything below is a case the fix must NOT change.
    ("a plain question keeps its figure", h("can you send me $20?"), "$20"),
    ("a genuine correction still wins",
     h("Dinner was $90 total", "sorry it was $120 actually"), "$120"),
    ("doubt with no refutation keeps the newer figure",
     h("tickets were $18 each", "I thought it was $15?"), "$15"),
    ("refutation with no doubt is untouched",
     h("send me $40", "No, I already paid you"), "$40"),
    ("an ask after a refuted doubt is still the ask",
     h("was it $15?", "No, that was the student rate", "can you send me $18?"), "$18"),
    ("two corrections in a row: the last wins",
     h("it was $90", "sorry $120", "actually $150 with the tip"), "$150"),
    ("a refusal that is not about the figure",
     h("can you send me $60 for the tickets", "No I'm broke"), "$60"),
]

# (name, [(sender, text)], expected amount on the prompt)
LIVE = [
    ("end to end: the refuted figure never reaches the sheet",
     [("20", "The museum tickets were $18 per person. Can everyone send me that?"),
      ("30", "Wait, I thought it was $15?"),
      ("20", "No, that was the student rate. We paid full price."),
      ("10", "ok sending")], "$18"),
    ("end to end: an ordinary ask is unaffected",
     [("20", "can you send me $20 for lunch?"), ("10", "sure sending")], "$20"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    a = ap.parse_args()
    failed = []

    for name, hist, want in UNIT:
        got = _latest_amount(hist)
        ok = got == want
        if not ok:
            failed.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"        want {want!r}, got {got!r}")

    if a.url:
        import requests
        tag = int(time.time())
        s = requests.Session()
        print()
        for ci, (name, seq, want) in enumerate(LIVE):
            room = f"group_am{tag}_{ci}"
            got = None
            fired = False
            for j, (sender, text) in enumerate(seq):
                d = s.post(a.url, json={"text": text, "room_id": room, "sender": sender,
                                        "participants": 4,
                                        "message_id": f"{tag}_{ci}_{j}"}, timeout=90).json()
                if [x for x in (d.get("intents") or []) if x in ("money", "ride")]:
                    fired = True
                    got = (d.get("slots") or {}).get("amount")
            ok = fired and got == want
            if not ok:
                failed.append(name)
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
            if not ok:
                print(f"        want {want!r}, got {got!r}"
                      + ("  (nothing fired)" if not fired else ""))

    total = len(UNIT) + (len(LIVE) if a.url else 0)
    print(f"\n  TOTAL {total - len(failed)}/{total} correct")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
