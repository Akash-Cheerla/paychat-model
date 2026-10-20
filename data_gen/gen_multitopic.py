"""Rooms where several things are being discussed at once, which is how rooms actually are.

Everything tested so far has been a room with one subject, or a day whose subjects happen
to arrive in blocks. Real rooms interleave: a money thread, a bug thread and lunch all
running in the same ten minutes, replies landing out of order against whichever one the
sender cares about.

That matters for the conversation summary specifically, because its points come out in
timestamp order. Three interleaved topics therefore produce a list that jumps between
them, and the questions are whether every topic survives, whether a minority topic gets
crowded out by a dominant one, and whether the near-duplicate collapse confuses two
topics that use similar words.

Each room is labelled per message with the topic it belongs to, so coverage is countable
rather than a judgement.

    python data_gen/gen_multitopic.py --out data/eval/multitopic.jsonl
"""
import argparse
import io
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Topic threads. Each is a list of (sender, text) in its own order; the generator
# interleaves them. Every line is something a summary SHOULD carry.
MONEY = [("13", "dinner last night came to 2400 total"),
         ("13", "can everyone send me 600 each"),
         ("70", "sending now"),
         ("26", "I need 650 for my rent this month, can you help out a friend"),
         ("13", "will send you back once i get paid on friday")]

BUG = [("4", "the split amount shows blank when participants is not sent"),
       ("4", "QA-116 still fails on the latest build"),
       ("20", "cannot set the group profile picture once the group is created"),
       ("4", "back swipe gesture not working in group chats"),
       ("20", "fixed the group admin issue, please verify on 61.3")]

RIDE = [("70", "can you book me a cab from my home to the airport"),
        ("70", "metro is stuck and my friend cant pick me up anymore"),
        ("13", "I need to leave by 4 so book it before that"),
        ("70", "pickup from JP Nagar works now")]

RELEASE = [("10", "the investor demo moved to Thursday 2pm"),
           ("10", "money flow and ride flow must both work end to end"),
           ("10", "ensure there are 0 issues in positive flow before the release"),
           ("10", "please read the notes and reply with objections before Thursday noon")]

CHATTER = [("26", "morning"), ("13", "morning"), ("70", "lol"), ("4", "haha yeah"),
           ("20", "ok"), ("26", "sure"), ("13", "nice"), ("70", "see you"),
           ("4", "ya"), ("20", "cool"), ("26", "thanks"), ("13", "hmm"),
           ("70", "right"), ("4", "done"), ("20", "yep"), ("26", "got it"),
           ("13", "the place downstairs is closed btw"),
           ("70", "the new one near the metro is decent"),
           ("4", "that storm is wild"), ("20", "lost power for an hour yesterday")]

THREADS = {"money": MONEY, "bug": BUG, "ride": RIDE, "release": RELEASE}


def interleave(threads, chatter, seed, chatter_every=2):
    """Round-robin the threads so no topic arrives in a block, with chatter between."""
    rnd = random.Random(seed)
    queues = {name: list(msgs) for name, msgs in threads.items()}
    chat = list(chatter)
    rnd.shuffle(chat)
    out = []
    while any(queues.values()):
        for name in list(queues):
            if not queues[name]:
                continue
            sender, text = queues[name].pop(0)
            out.append((sender, text, name))
            for _ in range(chatter_every):
                if chat:
                    s, t = chat.pop(0)
                    out.append((s, t, "chatter"))
    return out


def room(name, items, seed):
    base = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc) + timedelta(days=seed % 10)
    return [{"room": name, "sender": s,
             "ts": (base + timedelta(minutes=2 * i)).isoformat().replace("+00:00", "Z"),
             "text": t, "_topic": topic}
            for i, (s, t, topic) in enumerate(items)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/multitopic.jsonl")
    a = ap.parse_args()
    rows = []

    # 1. three topics fully interleaved, heavy chatter between
    rows += room("mt_three_interleaved",
                 interleave({k: THREADS[k] for k in ("money", "bug", "ride")},
                            CHATTER, 1), 1)
    # 2. all four at once - the worst case
    rows += room("mt_four_interleaved", interleave(THREADS, CHATTER, 2), 2)
    # 3. a topic switch: one subject, then a different one, no interleaving
    rows += room("mt_switch",
                 [(s, t, "money") for s, t in MONEY]
                 + [(s, t, "chatter") for s, t in CHATTER[:6]]
                 + [(s, t, "release") for s, t in RELEASE], 3)
    # 4. a dominant topic and a minority one - does the minority survive the ceiling?
    rows += room("mt_minority",
                 interleave({"bug": BUG * 3, "money": MONEY[:1]}, CHATTER, 4), 4)
    # 5. almost all chatter with two real items buried in it
    rows += room("mt_buried",
                 [(s, t, "chatter") for s, t in CHATTER]
                 + [(MONEY[0][0], MONEY[0][1], "money")]
                 + [(s, t, "chatter") for s, t in CHATTER[:8]]
                 + [(RIDE[2][0], RIDE[2][1], "ride")]
                 + [(s, t, "chatter") for s, t in CHATTER[8:]], 5)
    # 6. two topics that share vocabulary, to see if dedup confuses them
    rows += room("mt_similar",
                 interleave({"money": MONEY, "ride": RIDE}, CHATTER, 6), 6)

    path = ROOT / a.out
    with io.open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    per = collections.Counter((r["room"], r["_topic"]) for r in rows)
    print(f"  {len(rows)} messages, {len({r['room'] for r in rows})} rooms -> {a.out}")
    for rm in sorted({r["room"] for r in rows}):
        topics = {t: n for (r2, t), n in per.items() if r2 == rm}
        print(f"    {rm:<24} {sum(topics.values()):>3} msgs  {topics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
