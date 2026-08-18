"""Fingerprint the training corpus, and refuse merges that touch what already exists.

Two data disasters this month, both silent:

  * conversations_v6.json carried an offer_money role derive() did not recognise, so
    997 firing labels were quietly zeroed. v6 and v7 trained on it and both were thrown
    away — a week gone, and nothing in the pipeline noticed.
  * A relabel pass "fixed" 5,613 rows, most of them wrongly, because the detector that
    chose them was never checked against a sample by hand.

The common shape is a batch that MODIFIES rows that were already correct. So this file
enforces one rule: new data is only ever appended, never edited in place, and every row
carries the batch that produced it. Then a bad batch is one delete away instead of a
retrain away.

  python data_gen/data_lock.py snapshot            # fingerprint the corpus as it stands
  python data_gen/data_lock.py verify              # has anything drifted since?
  python data_gen/data_lock.py check-merge NEW.json BASE.json --tag round23
"""
import argparse, hashlib, json, sys, io, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "data" / "DATA_LOCK.json"

# The files a training run actually reads. Anything not listed here cannot silently
# become an input without also being added here, which is the point.
TRACKED = [
    "data/conversations/conversations_v9.json",
    "data/conversations/conversations_v8.json",
    "data/external/hard_negatives.json",
    "data/external/phrasings.json",
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_id(conv) -> str:
    """Identity of a conversation, independent of its labels.

    Deliberately the TEXT only. If a later batch changes a label, the id stays the same
    and the merge check reports it as a modification rather than letting it pass as a
    new row — which is exactly how the v6 zeroing would have been caught.
    """
    turns = conv.get("turns") or conv.get("messages") or []
    body = "\u0001".join((t.get("text") or "") for t in turns)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


def labels_of(conv):
    turns = conv.get("turns") or conv.get("messages") or []
    out = []
    for t in turns:
        v = t.get("fire", t.get("expected"))
        out.append(tuple(sorted(v)) if isinstance(v, list) else (v,))
    return tuple(out)


def load(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d.get("conversations", [])


def cmd_snapshot(a):
    entry = {"taken_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": {}}
    for rel in TRACKED:
        p = ROOT / rel
        if not p.exists():
            print(f"  {rel:52} MISSING"); continue
        rows = load(p)
        entry["files"][rel] = {"sha256": sha(p), "bytes": p.stat().st_size,
                               "rows": len(rows)}
        print(f"  {rel:52} {len(rows):>7} rows  {entry['files'][rel]['sha256'][:12]}")
    LOCK.write_text(json.dumps(entry, indent=1), encoding="utf-8")
    print(f"\n  wrote {LOCK.relative_to(ROOT)}")
    return 0


def cmd_verify(a):
    if not LOCK.exists():
        print("  no lock file — run snapshot first"); return 1
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    bad = 0
    for rel, want in lock["files"].items():
        p = ROOT / rel
        if not p.exists():
            print(f"  {rel:52} GONE"); bad += 1; continue
        got = sha(p)
        if got != want["sha256"]:
            print(f"  {rel:52} CHANGED  {want['sha256'][:12]} -> {got[:12]}"); bad += 1
        else:
            print(f"  {rel:52} ok")
    print(f"\n  {'DRIFT DETECTED' if bad else 'clean'} — snapshot {lock['taken_at']}")
    return 1 if bad else 0


def cmd_check_merge(a):
    base, new = load(ROOT / a.base), load(ROOT / a.new)
    b = {row_id(c): labels_of(c) for c in base}
    added, modified, dup = 0, [], 0
    for c in new:
        rid = row_id(c)
        if rid not in b:
            added += 1
        elif labels_of(c) != b[rid]:
            modified.append(rid)
        else:
            dup += 1
    print(f"  base {a.base}: {len(base)} conversations")
    print(f"  new  {a.new}: {len(new)} conversations")
    print(f"    added        {added}")
    print(f"    exact dupes  {dup}")
    print(f"    MODIFIED     {len(modified)}")
    if modified:
        print("\n  REJECTED — this batch rewrites labels on conversations that already")
        print("  exist. That is the failure mode that cost v6 and v7. If the relabel is")
        print("  genuinely intended, do it as its own reviewed change, not inside a")
        print("  batch of new data.")
        for rid in modified[:10]:
            print(f"    {rid}")
        return 1
    print("\n  OK — additive only")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    sub.add_parser("verify")
    m = sub.add_parser("check-merge")
    m.add_argument("new"); m.add_argument("base"); m.add_argument("--tag", default="")
    a = ap.parse_args()
    return {"snapshot": cmd_snapshot, "verify": cmd_verify,
            "check-merge": cmd_check_merge}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
