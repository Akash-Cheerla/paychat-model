"""Realistic chat, written the way these rooms actually read, with two or three topics.

The earlier multi-topic corpus was skeleton lines - one clean sentence per topic with
"ok"/"lol" between. Real rooms are not like that: people half-answer, correct themselves
two messages later, ask something unrelated mid-thread, send three fragments instead of
one sentence, and the useful content arrives inside the mess rather than instead of it.

That matters for grouping, because the question is whether a summary can separate the
threads when they are genuinely tangled rather than neatly alternating.

Three rooms:
    group_release   a work group, three threads - a build to test, a dinner split,
                    and a demo date moving
    dm_cab_money    a 1v1 where a cab and a repayment are negotiated at the same time
    group_flat      a flatmates group, two threads - rent/bills and a plumber visit

Everything here is invented, so it can be committed and shared, unlike the WhatsApp
exports.

    python data_gen/gen_realistic_rooms.py --out data/eval/realistic_rooms.jsonl
"""
import argparse
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (sender, text, topic) - topic is the label, used only to score grouping afterwards.
GROUP_RELEASE = [
    ("yash", "morning all", "chatter"),
    ("brahma", "morning", "chatter"),
    ("yash", "build 61.3 is up on testflight", "build"),
    ("yash", "please pull it before you test anything today", "build"),
    ("rupesh", "pulling now", "chatter"),
    ("brahma", "is the group profile pic thing in this one?", "build"),
    ("yash", "yeah that's fixed", "build"),
    ("yash", "also the back swipe gesture in group chats", "build"),
    ("brahma", "nice", "chatter"),
    ("andril", "guys dinner last night came to 2400 total", "dinner"),
    ("andril", "there were 4 of us so 600 each", "dinner"),
    ("rupesh", "sending now", "dinner"),
    ("brahma", "i'll do it tonight", "dinner"),
    ("gowtham", "quick one - the investor demo moved up", "demo"),
    ("gowtham", "it's thursday 2pm now instead of next monday", "demo"),
    ("yash", "oh", "chatter"),
    ("gowtham", "so money and ride both need to work end to end by then", "demo"),
    ("andril", "actually wait", "dinner"),
    ("andril", "rupesh paid for the drinks separately so his share is 450 not 600", "dinner"),
    ("rupesh", "yeah i already sent 600 though", "dinner"),
    ("andril", "i'll send you 150 back", "dinner"),
    ("brahma", "lol", "chatter"),
    ("yash", "QA-116 is still failing on 61.3 btw", "build"),
    ("yash", "the split amount shows blank when participants isn't sent", "build"),
    ("brahma", "that's mine, looking", "build"),
    ("gowtham", "is that a blocker for thursday?", "demo"),
    ("yash", "shouldn't be, it only happens without participants", "build"),
    ("gowtham", "ok", "chatter"),
    ("gowtham", "let's sync at 4 today and again thursday morning", "demo"),
    ("rupesh", "works for me", "chatter"),
    ("brahma", "same", "chatter"),
    ("andril", "cab booking from current location works now too", "build"),
    ("yash", "confirmed", "chatter"),
]

DM_CAB_MONEY = [
    ("13", "hey", "chatter"),
    ("70", "yo", "chatter"),
    ("13", "can you book me a cab", "cab"),
    ("70", "from where", "cab"),
    ("13", "home to the airport", "cab"),
    ("70", "what time", "cab"),
    ("13", "i need to leave by 4", "cab"),
    ("70", "ok booking", "cab"),
    ("13", "also", "money"),
    ("13", "did you get the 500 i sent last week", "money"),
    ("70", "yeah got it", "money"),
    ("70", "i still owe you 300 from the dinner though", "money"),
    ("13", "no rush", "chatter"),
    ("70", "i'll send it friday when i get paid", "money"),
    ("13", "cool", "chatter"),
    ("70", "actually the cab says 40 mins away", "cab"),
    ("13", "that's too late", "cab"),
    ("13", "can you try a different one", "cab"),
    ("70", "trying", "chatter"),
    ("70", "got one, 12 mins", "cab"),
    ("13", "perfect thanks", "chatter"),
    ("70", "it's 680 for the ride", "cab"),
    ("13", "put it on my tab with the 300", "money"),
    ("70", "so i owe you 300 and you owe me 680", "money"),
    ("70", "net you owe me 380", "money"),
    ("13", "yeah that works", "chatter"),
    ("13", "sending 380 now", "money"),
]

GROUP_FLAT = [
    ("a", "guys rent is due on the 5th", "rent"),
    ("b", "how much is it this month", "rent"),
    ("a", "same, 18000 split three ways so 6000 each", "rent"),
    ("c", "ok", "chatter"),
    ("b", "the electricity bill came too", "rent"),
    ("b", "2400 for last month", "rent"),
    ("a", "that's high", "chatter"),
    ("c", "the ac was on all night most days tbh", "chatter"),
    ("a", "fair", "chatter"),
    ("c", "btw the kitchen tap is leaking again", "plumber"),
    ("c", "it's getting worse", "plumber"),
    ("a", "i'll call the plumber", "plumber"),
    ("b", "the same guy from last time?", "plumber"),
    ("a", "yeah", "plumber"),
    ("a", "he said he can come saturday morning", "plumber"),
    ("c", "i'm out saturday", "chatter"),
    ("b", "i'll be here", "plumber"),
    ("a", "ok i'll tell him saturday 10am", "plumber"),
    ("b", "so 6000 for rent plus 800 for electricity each?", "rent"),
    ("a", "yes 6800 each", "rent"),
    ("c", "sending", "chatter"),
    ("b", "i'll send tomorrow", "rent"),
    ("a", "also the plumber wants 600 for the visit", "plumber"),
    ("a", "we can split that too", "plumber"),
    ("b", "200 each then", "plumber"),
    ("c", "fine", "chatter"),
]

ROOMS = {
    "group_release": (GROUP_RELEASE, datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)),
    "dm_cab_money": (DM_CAB_MONEY, datetime(2026, 11, 12, 14, 0, tzinfo=timezone.utc)),
    "group_flat": (GROUP_FLAT, datetime(2026, 11, 13, 19, 0, tzinfo=timezone.utc)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/realistic_rooms.jsonl")
    a = ap.parse_args()
    rows = []
    for room, (msgs, base) in ROOMS.items():
        for i, (sender, text, topic) in enumerate(msgs):
            rows.append({"room": room, "sender": sender,
                         "ts": (base + timedelta(minutes=2 * i + (i % 3))
                                ).isoformat().replace("+00:00", "Z"),
                         "text": text, "_topic": topic})
    with io.open(ROOT / a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(f"  {len(rows)} messages, {len(ROOMS)} rooms -> {a.out}")
    for room in ROOMS:
        c = collections.Counter(r["_topic"] for r in rows if r["room"] == room)
        print(f"    {room:<16} {sum(c.values()):>3} msgs  {dict(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
