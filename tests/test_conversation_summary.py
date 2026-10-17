"""Does the room summary surface what a day contained, and refuse what it didn't?

The case this feature exists for is dm_13_70 on 2026-08-30: 323 messages, every one of
them too short for a per-message card, so the Summary tab shows NOTHING - while the day
is a money-and-ride thread from end to end. The rules on the whole window pull out twelve
points from it.

What is asserted here is mostly the refusals, because that is where a conversation summary
does damage. It reads every message in a room, including the ones nobody would want
quoted, so the filters matter more than the recall:

  stands alone      "Can you bring it up?" scores the same 3 as "Ohh btw can you send me
                    500 ?" - both match the request pattern - but the reader cannot act on
                    the first without scrolling back to find what "it" was, which is the
                    thing a summary exists to save them.
  near-duplicate    a room says "Can you book cab for me" fifteen times.
  quiet windows      draw no card at all, the same rule a short message follows.
  no invented text   every point is a span of something somebody said, carried with its
                    sender and timestamp. There is no generation here and so no way for a
                    point to say something nobody did - which is the whole reason the
                    engine is the rules and not the student.

    python tests/test_conversation_summary.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import room_summarizer as R   # noqa: E402

# Found by auditing all 253 summaries the logs produce, before the feature was shown to
# anyone. This path reads EVERY message in a room, which is a different exposure from the
# per-message card, and the corpus that exposed it is the anonymised WhatsApp exports -
# real human conversation, and the right population to check for a consumer chat app.
SENSITIVE = [
    ("an OTP request never reaches a summary",
     "Can you please share the OTP for dejavoo portal please", True),
    ("a phone number never reaches a summary",
     "Hi sir can you add this number in wisecats group 7320863304", True),
    ("a password never reaches a summary",
     "We also can't change the wifi password", True),
    ("a UPI reference never reaches a summary",
     "Failed because upi is not set on your end", True),
    ("a card-shaped number never reaches a summary",
     "use 4111 1111 1111 for the test", True),
    ("an ordinary money request is not sensitive",
     "Can you send me 500 for the cab", False),
    ("a deadline is not sensitive", "I need to leave by 4", False),
]

PRIVATE = [
    ("personal finance news is refused",     "My salary got delayed", True),
    ("a social ask is refused",              "Give me a few too, will you smoke with me?", True),
    ("substance chatter is refused",         "Let me smoke one cig, it's awesome", True),
    ("health is refused",                    "I'm sick today, working from home", True),
    # The rescue matters more than the refusal: a money request that happens to mention
    # rent or salary is the thing this feature exists for.
    ("a rent request survives",              "I need 650$ for my rent this month, can you help out a friend?", False),
    ("a repayment promise survives",         "Will send you back once i got salary", False),
    ("a split that mentions drinks survives","beers were 800 split 4 ways", False),
    ("a cab booking survives",               "book a cab from home to office", False),
]

OPENERS = [
    ("a leading But is stripped",   "But june's is still due", "June's is still due"),
    ("a leading So is stripped",    "So need something asap", "Need something asap"),
    ("a leading And also is stripped", "And also check if filter is blocked", "Check if filter is blocked"),
    ("stacked fillers are stripped", "Ohh btw can you send me 500 ?", "Can you send me 500 ?"),
    ("a clean point is untouched",  "Metro is stuck", "Metro is stuck"),
    ("stripping stops before it guts the point", "But same issue", "But same issue"),
]


def msg(i, sender, text, hour=9, minute=0):
    return {"sender": sender, "text": text,
            "ts": f"2026-10-16T{hour:02d}:{minute:02d}:{i:02d}Z"}


# The real thread, shortened. Every message is too short for a per-message card.
MONEY_RIDE = [
    msg(1, "13", "hey"), msg(2, "70", "yo"),
    msg(3, "13", "Ohh btw can you send me 500 ?"),
    msg(4, "13", "Actually nvm can you just send 300 so i can close this out"),
    msg(5, "70", "Metro is stuck"),
    msg(6, "70", "Can you book a cab to the office ?"),
    msg(7, "70", "My friend can't pick me up anymore"),
    msg(8, "13", "haha"),
    msg(9, "70", "Im literally at 200 until Friday"),
    msg(10, "70", "I need to leave by 4"),
    msg(11, "13", "ok"), msg(12, "13", "sure"),
]

CHATTER = [msg(i, "a" if i % 2 else "b", t) for i, t in enumerate([
    "morning", "morning", "lol", "haha yeah", "ok", "sure", "nice", "see you",
    "ha", "yep", "cool", "thanks"], 1)]

REPEATS = [msg(1, "13", "hey"), msg(2, "70", "hi")] + [
    msg(i, "13", "Can you book cab for me") for i in range(3, 12)]

# A pronoun-only exchange: real conversation, nothing a summary can carry.
ANAPHORA = [msg(1, "13", "hey"), msg(2, "70", "yo")] + [
    msg(3, "70", "Can you bring it up?"), msg(4, "13", "Already did can't find it"),
    msg(5, "70", "Actually we can move it?"), msg(6, "13", "ok"),
    msg(7, "70", "Actually can you snag one for me ?"), msg(8, "13", "sure"),
    msg(9, "70", "did you?"), msg(10, "13", "yes"), msg(11, "70", "ok"),
    msg(12, "13", "cool")]

STANDS_ALONE = [
    ("a bare pronoun object is refused",            "Can you bring it up?", False),
    ("an anaphoric reply is refused",               "Already did can't find it", False),
    ("a pronoun-only proposal is refused",          "Actually we can move it?", False),
    ("'one' with no referent is refused",           "Actually can you snag one for me ?", False),
    ("a number names the thing",                    "Ohh btw can you send me 500 ?", True),
    ("a concrete noun names the thing",             "Can you book a cab to the office ?", True),
    ("'this one on your CARD' is concrete",         "Can you put this one on your card?", True),
    ("a deadline stands alone",                     "I need to leave by 4", True),
    ("a proper noun mid-sentence names the thing",  "Can you ask Priya about the spec?", True),
    ("a ticket id stands alone",                    "QA-116 still fails", True),
    ("a bare statement stands alone",               "Metro is stuck", True),
]


def main():
    failed = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed.append(name)
            print(f"        got {got!r}, want {want!r}")

    print("\n  a point has to stand on its own")
    for name, text, want in STANDS_ALONE:
        check(name, R._stands_alone(text), want)

    print("\n  the window as a whole")
    s = R.summarize_room(MONEY_RIDE, room="dm_13_70")
    check("a day of short messages produces a summary", s is not None, True)
    if s:
        joined = " | ".join(p["text"] for p in s.points)
        check("the amount revision survives", "300" in joined, True)
        check("the ride request survives", "cab" in joined.lower(), True)
        check("the deadline survives", "leave by 4" in joined, True)
        check("pleasantries do not", "haha" in joined.lower(), False)
        check("points carry a sender", all(p.get("sender") for p in s.points), True)
        check("points carry a timestamp", all(p.get("ts") for p in s.points), True)
        check("points are oldest first",
              [p["ts"] for p in s.points] == sorted(p["ts"] for p in s.points), True)
        # The anti-fabrication property, and the reason the engine is the rules. Points
        # are no longer literal substrings - _tidy strips a leading conjunction and
        # recapitalises - so what is asserted is that no point contains a WORD nobody
        # said. A generated summary cannot pass this; "Meeting Rescheduled" did not.
        import re as _re
        said = set(_re.findall(r"[a-z0-9']+", " ".join(m["text"] for m in MONEY_RIDE).lower()))
        invented = [w for p in s.points
                    for w in _re.findall(r"[a-z0-9']+", p["text"].lower())
                    if w not in said]
        check("no point contains a word nobody said", invented, [])
        check("the window is reported",
              bool(s.window_from and s.window_to and s.window_from <= s.window_to), True)
        check("shape counts what is in it", bool(s.shape), True)
        check("never more than the ceiling", len(s.points) <= R.MAX_POINTS, True)

    check("a window of pure chatter draws no card",
          R.summarize_room(CHATTER, room="x") is None, True)
    check("a window of pronouns draws no card",
          R.summarize_room(ANAPHORA, room="x") is None, True)
    check("a window below the message floor draws no card",
          R.summarize_room(MONEY_RIDE[:4], room="x") is None, True)
    check("no messages at all draws no card", R.summarize_room([], room="x") is None, True)

    rep = R.summarize_room(REPEATS, room="x")
    check("nine copies of one request collapse",
          rep is None or sum(1 for p in rep.points if "cab" in p["text"].lower()) <= 2, True)

    print("\n  the window helper")
    check("since excludes what came before",
          len(R.window(MONEY_RIDE, since="2026-10-16T09:00:06Z")) < len(MONEY_RIDE), True)
    check("no window argument keeps everything",
          len(R.window(MONEY_RIDE)) == len(MONEY_RIDE), True)
    check("a lookback keeps the recent end",
          len(R.window(MONEY_RIDE, hours=1)) == len(MONEY_RIDE), True)
    check("a malformed timestamp does not raise",
          isinstance(R.window([{"text": "hi", "ts": "not-a-date", "sender": "a"}],
                              hours=2), list), True)

    total = len(STANDS_ALONE) + len(SENSITIVE) + len(PRIVATE) + len(OPENERS) + 20
    print(f"\n  TOTAL {total - len(failed)}/{total} correct")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
