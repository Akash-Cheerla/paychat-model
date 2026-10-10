"""Clean the teacher's output into a training set that teaches the right thing.

Three defects in the generated pairs, all fixable here rather than by re-running the GPU:

  PREAMBLE   11% of messages open with the teacher talking to us - "here's a message that
             fits your criteria:", "here we go:". A student trained on these learns to
             expect a preamble that will never exist in production. The strip in
             distill_pairs.py was too narrow; this one handles the forms actually seen.

  CHATTER    319 of 356 messages generated AS chatter got a card anyway, leaving 37 empty
             cards in 1,796 (2%, designed for 20%). Trained on that, the student learns to
             always produce a card - which is precisely the failure Akash rejected on
             sight in the iOS screenshots. We know the ground truth by construction: the
             chatter prompt asked for a message with nothing to remember, so the card is
             forced empty rather than trusting the teacher's second opinion.

  FILLER     points that carry no fact - "We appreciate your help", "Let us know how to
             assist". A summary card is a thing to act on, not a pleasantry.

Deliberately NOT fixed here: points per card came out at 4.7 against 6.9 on the labelled
set. That is the teacher being less thorough on its own inventions than on real-looking
messages, and no amount of cleaning adds a point that was never written.

    python data_gen/clean_distill_pairs.py --in <downloaded>.jsonl --out data/distill/pairs.jsonl
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# "here's a message that fits your criteria:", "here we go:", "sure, here is one:"
PREAMBLE = re.compile(
    r"^\s*(?:sure|certainly|of course|okay|ok|alright|here\s+we\s+go|here'?s?|"
    r"below\s+is|this\s+is|the\s+message\s+is)\b[^:\n]{0,90}:\s*", re.IGNORECASE)

# A point that says nothing to act on.
FILLER = re.compile(
    r"^\s*(?:we\s+appreciate|thanks?\s|thank\s+you|let\s+us\s+know|let\s+me\s+know|"
    r"please\s+(?:let|feel\s+free|advise)|provide\s+additional\s+details|"
    r"reach\s+out|looking\s+forward|happy\s+to\s+help|no\s+rush|hope\s+(?:you|this)|"
    r"asked\s+about|asked\s+if|considering\s+options?|.{0,20}\bunknown\s*$)",
    re.IGNORECASE)


def strip_preamble(text):
    cleaned = PREAMBLE.sub("", text or "", count=1).lstrip()
    # the preamble is often on its own line, with the message after a blank line
    return cleaned.lstrip("\n").strip()


def clean(rows, force_chatter_empty=True):
    out, stats = [], {}

    def bump(k):
        stats[k] = stats.get(k, 0) + 1

    for r in rows:
        msg = strip_preamble(r["message"])
        if msg != r["message"]:
            bump("preamble stripped")
        if len(msg.split()) < 15:
            bump("dropped: too short after stripping")
            continue

        card = dict(r.get("card") or {})
        points = [p for p in (card.get("points") or []) if isinstance(p, str)]

        if r.get("chatter") and force_chatter_empty:
            if points:
                bump("chatter card emptied")
            card = {"title": None, "points": []}
        else:
            kept = [p for p in points if not FILLER.match(p)]
            if len(kept) != len(points):
                bump("filler point removed")
            if points and not kept:
                bump("dropped: every point was filler")
                continue
            card["points"] = kept
            if kept and not card.get("title"):
                bump("dropped: points without a title")
                continue

        out.append({**r, "message": msg, "card": card})
    return out, stats


def report(rows, label):
    carded = [r for r in rows if r["card"].get("points")]
    npts = sum(len(r["card"]["points"]) for r in carded)
    pre = sum(1 for r in rows if PREAMBLE.match(r["message"]))
    print(f"  {label:<10} {len(rows):>5} pairs | {len(rows)-len(carded):>4} empty "
          f"({100*(len(rows)-len(carded))/max(len(rows),1):>4.1f}%) | "
          f"{npts/max(len(carded),1):.1f} pts/card | {pre} with preamble")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", default="data/distill/pairs.jsonl")
    ap.add_argument("--keep-chatter-cards", action="store_true",
                    help="trust the teacher on chatter instead of the generation spec")
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.src).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    report(rows, "before")
    out, stats = clean(rows, force_chatter_empty=not a.keep_chatter_cards)
    report(out, "after")

    print()
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"    {v:>5}  {k}")

    dest = ROOT / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  -> {a.out}  ({dest.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
