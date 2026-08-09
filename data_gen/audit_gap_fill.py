"""Audit the gap-fill training data before it is merged into a training set.

Written because the last two data problems both got caught late: group_llm_v5 turned
out to be 100% inside the training source (so the group eval measured nothing), and
the first behaviour_eval run had rows whose wording contradicted their label. Both
would have been caught by running this first.

Fails loudly. Anything it reports as CONTAMINATION or LABEL is a stop-ship.

Run:  python data_gen/audit_gap_fill.py data/conversations/gap_fill_v6.json [more...]
"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EVAL_FILES = ["data/eval/group_eval_v6.json", "data/eval/behaviour_eval.json",
              "data/eval/claude_eval.json", "data/eval/eval_conversations.json",
              "data/eval/mixed_eval.json"]
TRAIN_FILES = ["data/conversations/conversations_v5.json"]

HIN = ("bhai", "yaar", "paisa", "kitna", "acha", "theek", "haan", "nahi", "kya", "arre")
# A turn labelled to fire must not be reporting something already finished (rule 3a)
# or promising it for later (rule 3).
DONE = re.compile(r"\b(just (sent|paid|booked)|already (sent|paid|booked|did)|"
                  r"sent it|paid it|have sent)\b", re.I)
FUTURE = re.compile(r"\b(tomorrow|tmrw|next week|on (mon|tues|wednes|thurs|fri|satur|sun)day|"
                    r"later tonight|after i get|by tomorrow|next month)\b", re.I)


def turns_of(c):
    return c.get("turns") or c.get("messages") or []


def sig(c):
    return hashlib.md5("|".join(t.get("text", "") for t in turns_of(c))
                       .encode("utf-8", "replace")).hexdigest()


def load(p):
    f = ROOT / p
    if not f.exists():
        return []
    d = json.loads(f.read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d.get("conversations", [])


def main(paths):
    rows = []
    for p in paths:
        rows += load(p)
    if not rows:
        sys.exit("no conversations loaded")
    print(f"auditing {len(rows)} conversations from {len(paths)} file(s)\n")

    fails = 0

    # 1. contamination — the failure that made the group eval meaningless
    sigs = [sig(c) for c in rows]
    for label, files in (("EVAL", EVAL_FILES), ("TRAIN", TRAIN_FILES)):
        other = set()
        for p in files:
            other |= {sig(c) for c in load(p)}
        hits = sum(1 for s in sigs if s in other)
        tag = "CONTAMINATION" if (hits and label == "EVAL") else "ok"
        print(f"  [{tag:13}] overlap with {label:5}: {hits}")
        fails += (hits > 0 and label == "EVAL")

    dupes = len(sigs) - len(set(sigs))
    print(f"  [{'ok' if not dupes else 'DUPES':13}] internal duplicates: {dupes}")

    # 2. label sanity — the failure that made behaviour_eval's first run unusable
    bad_done = bad_future = bad_empty = 0
    for c in rows:
        for t in turns_of(c):
            if not t.get("fire"):
                continue
            if DONE.search(t.get("text", "")):
                bad_done += 1
            if FUTURE.search(t.get("text", "")):
                bad_future += 1
            if not t.get("text", "").strip():
                bad_empty += 1
    for name, n in (("fires but already done", bad_done),
                    ("fires but future-dated", bad_future),
                    ("fires but empty text", bad_empty)):
        print(f"  [{'LABEL' if n else 'ok':13}] {name}: {n}")
        fails += n > 0

    # 3. language
    hin = sum(1 for c in rows for t in turns_of(c)
              if any(h in t.get("text", "").lower().split() for h in HIN))
    print(f"  [{'HINGLISH' if hin else 'ok':13}] Hinglish turns: {hin}")
    fails += hin > 0

    # 4. balance — a set that only ever fires teaches the model to always fire
    tt = sum(len(turns_of(c)) for c in rows)
    ff = sum(1 for c in rows for t in turns_of(c) if t.get("fire"))
    neg = sum(1 for c in rows if not any(t.get("fire") for t in turns_of(c)))
    print(f"\n  turns {tt}, firing {ff} ({ff/tt:.1%})")
    print(f"  conversations that fire nothing: {neg} ({neg/len(rows):.1%})")
    if neg / len(rows) < 0.10:
        print("  [BALANCE     ] under 10% negatives — the model will over-fire")
        fails += 1

    # 5. diversity — recombined text overfits surface form
    texts = [t.get("text", "").lower().strip() for c in rows for t in turns_of(c)]
    uniq = len(set(texts))
    print(f"  unique turn texts: {uniq}/{len(texts)} ({uniq/len(texts):.1%})")
    if uniq / len(texts) < 0.60:
        print("  [DIVERSITY    ] under 60% unique — too repetitive")
        fails += 1

    print(f"\n  by scenario: {dict(Counter(c.get('scenario','?') for c in rows))}")
    print(f"  by speakers: {dict(sorted(Counter(len({t['sender'] for t in turns_of(c)}) for c in rows).items()))}")
    print("\n" + ("  AUDIT PASSED" if not fails else f"  AUDIT FAILED — {fails} problem(s)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["data/conversations/gap_fill_v6.json"]))
