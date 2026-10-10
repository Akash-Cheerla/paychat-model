"""PROTOTYPE: summarise a ROOM OVER A DAY instead of one message at a time.

Akash, 2026-10-08: per-message cards only see 2.9% of traffic, while 9.9% carries a
deadline, instruction, decision or problem in messages too short to ever qualify - and
isn't money or ride either, so neither tab shows it. A decision reached across four short
messages is invisible to everything we have.

This changes exactly ONE thing against the shipped summarizer: the UNIT. Same scoring,
same vocabulary, same chatter rules - applied to every message in a room for a day rather
than to single long messages. Holding the engine fixed is the point: if the result is
useful, the unit was the problem; if it is not, no model was going to save it either.

Runs entirely locally. group_12 is a real private team chat and nothing here sends it
anywhere.

    python data_gen/proto_room_summary.py --room group_12 --day 2026-10-01
"""
import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import summarizer as S          # noqa: E402

# A day of a working group is mostly repetition. The dogfood log is the extreme case -
# "Can you book cab for me" fifteen times - but any real room repeats itself.
NEAR_DUPLICATE = 0.75


def content(text):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", (text or "").lower())
            if w not in S._FUNCTION_WORDS and len(w) > 2}


def is_near_duplicate(words, seen):
    if not words:
        return True
    for prev in seen:
        if not prev:
            continue
        overlap = len(words & prev) / max(len(words | prev), 1)
        if overlap >= NEAR_DUPLICATE:
            return True
    return False


def day_summary(messages, limit=12):
    """The things worth remembering from one day in one room, newest-last.

    Each message is scored whole - short ones included, which is the entire point. A
    message long enough for its own card is first cut into points so that a nine-sentence
    status update contributes its deadline rather than its throat-clearing.
    """
    scored = []
    for m in messages:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        if S.is_worth_summarizing(text):
            # Long message: take the points its own card would have shown.
            card = S.summarize(text)
            candidates = card.points if card else []
        else:
            candidates = [S._tidy_point(text)]
        for c in candidates:
            if not c or not S._is_substantive(c):
                continue
            score = S.importance(c)
            if score < 2:                 # same bar a card has to clear
                continue
            scored.append({"sender": m.get("sender"), "ts": m.get("ts"),
                           "text": c, "score": score})

    # Strongest first, so a duplicate is dropped in favour of the better-scoring copy.
    scored.sort(key=lambda s: (-s["score"], s["ts"] or ""))
    kept, seen = [], []
    for s in scored:
        words = content(s["text"])
        if is_near_duplicate(words, seen):
            continue
        seen.append(words)
        kept.append(s)
        if len(kept) >= limit:
            break
    kept.sort(key=lambda s: s["ts"] or "")
    return kept


def day_heading(points) -> str:
    """One heading for the day, from the points themselves.

    The same title_for() a per-message card uses - for the purpose of naming it, a day of
    a room is just a longer message.
    """
    if not points:
        return "Nothing to report"
    return S.title_for(" . ".join(p["text"] for p in points))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/eval/dogfood_2026-10-07.jsonl")
    ap.add_argument("--room", default="group_12")
    ap.add_argument("--day", default="2026-10-01", help="first day")
    ap.add_argument("--to", help="last day of a range")
    ap.add_argument("--all-rooms", action="store_true")
    ap.add_argument("--min-messages", type=int, default=12,
                    help="skip a room/day quieter than this")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--compare", action="store_true",
                    help="also show what the per-message cards gave for the same day")
    a = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / a.log).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    days = sorted({(r.get("ts") or "")[:10] for r in rows if r.get("ts")})
    days = [d for d in days if d >= a.day and (not a.to or d <= a.to)]

    buckets = defaultdict(list)
    for r in rows:
        day = (r.get("ts") or "")[:10]
        if day not in days:
            continue
        if not a.all_rooms and r.get("room") != a.room:
            continue
        buckets[(day, r.get("room"))].append(r)

    summaries = total_pts = total_msgs = 0
    for day in days:
        keys = sorted((k for k in buckets if k[0] == day), key=lambda k: -len(buckets[k]))
        header_shown = False
        for key in keys:
            msgs = buckets[key]
            if len(msgs) < a.min_messages:
                continue
            out = day_summary(msgs, a.limit)
            if not out:
                continue
            if not header_shown:
                print(f"\n\n  {'#'*76}\n  {day}\n  {'#'*76}")
                header_shown = True
            summaries += 1
            total_pts += len(out)
            total_msgs += len(msgs)
            print(f"\n  {key[1]}  -  {day_heading(out)}"
                  f"   ({len(msgs)} messages -> {len(out)} points)")
            print(f"  {'-'*76}")
            for s in out:
                print(f"   {s['ts'][11:16]} [{s['sender']:>2}]  {s['text'][:86]}")

    print(f"\n\n  {'='*78}")
    print(f"  {summaries} summaries over {len(days)} days: "
          f"{total_pts} points from {total_msgs} messages")

    if a.compare:
        msgs = [r for r in rows if (r.get("ts") or "")[:10] in days
                and (a.all_rooms or r.get("room") == a.room)]
        cards = [(m, S.summarize(m.get('text') or '')) for m in msgs]
        cards = [(m, c) for m, c in cards if c]
        print(f"\n  what the PER-MESSAGE tab shows for the same day: {len(cards)} card(s)")
        for m, c in cards:
            print(f"    {m['ts'][11:16]} [{m['sender']}] {c.title} "
                  f"({len(c.points)} bullets)")
        covered = sum(1 for m in msgs if S.is_worth_summarizing(m.get('text') or ''))
        print(f"    reaching {covered} of {len(msgs)} messages "
              f"({100*covered/len(msgs):.0f}% of the day)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
