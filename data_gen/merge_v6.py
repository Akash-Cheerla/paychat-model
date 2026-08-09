"""Merge the gap-fill data into a v6 training set.

Takes conversations_v5 (what the shipped model was trained on) and adds the three
gap-fill sets generated 2026-08-09. Drops the superseded single-payer splits from
gap_fill_v6 — those were generated before the multi-payer bug was found, and leaving
them in would teach "one person pays a split" alongside the corrected "everyone pays",
which is the behaviour the round exists to fix.

Re-runs the contamination check on the OUTPUT, not just the inputs. That is the check
that would have caught group_llm_v5 sitting 100% inside the training source, which is
why the group eval measured nothing for a whole training round.

Run:  python data_gen/merge_v6.py
"""
import hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE = "data/conversations/conversations_v5.json"
ADDS = ["data/conversations/gap_fill_v6.json",
        "data/conversations/gap_fill_offerq.json",
        "data/conversations/gap_fill_splits.json"]
OUT = "data/conversations/conversations_v6.json"
EVALS = ["data/eval/group_eval_v6.json", "data/eval/behaviour_eval.json",
         "data/eval/claude_eval.json", "data/eval/eval_conversations.json",
         "data/eval/mixed_eval.json"]


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


def main():
    base = load(BASE)
    print(f"base {BASE}: {len(base)}")

    add = []
    for p in ADDS:
        rows = load(p)
        if "gap_fill_v6" in p:
            # its splits predate the multi-payer fix
            before = len(rows)
            rows = [c for c in rows if c.get("scenario") != "gap_big_split"]
            print(f"add  {p}: {len(rows)}  (dropped {before - len(rows)} superseded splits)")
        else:
            print(f"add  {p}: {len(rows)}")
        add += rows

    seen = {sig(c) for c in base}
    fresh, dupe = [], 0
    for c in add:
        s = sig(c)
        if s in seen:
            dupe += 1
            continue
        seen.add(s)
        fresh.append(c)
    merged = base + fresh
    print(f"\nadded {len(fresh)} new ({dupe} already present)")
    print(f"total {len(merged)}")

    ev = set()
    for p in EVALS:
        ev |= {sig(c) for c in load(p)}
    leak = sum(1 for c in merged if sig(c) in ev)
    print(f"\nCONTAMINATION against {len(EVALS)} eval sets: {leak}")
    if leak:
        sys.exit("REFUSING TO WRITE — training set overlaps the evals")

    spk = Counter(len({t.get("sender") for t in turns_of(c)}) for c in merged)
    grp = sum(v for k, v in spk.items() if k >= 3)
    big = sum(v for k, v in spk.items() if k >= 4)
    tt = sum(len(turns_of(c)) for c in merged)
    ff = sum(1 for c in merged for t in turns_of(c) if t.get("fire"))
    print(f"\nspeakers   : {dict(sorted(spk.items()))}")
    print(f"group share: {grp}/{len(merged)} = {grp/len(merged):.1%}  (4+ speakers: {big})")
    print(f"turns      : {tt}, firing {ff} ({ff/tt:.1%})")

    (ROOT / OUT).write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
