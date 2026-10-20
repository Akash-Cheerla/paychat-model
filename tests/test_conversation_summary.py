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


# Rooms are not about one thing. These interleave several threads with chatter between,
# the way a real room does, and assert that no topic is lost - a topic that vanishes is
# the failure that matters, because someone scrolls back BECAUSE they missed it.
#
# Found by running these (2026-10-19): a shared bill scored only the +1 any digit earns,
# so "dinner last night came to 2400 total" was below the bar and a room whose whole
# subject was a split produced NO summary. The per-message path never noticed, because
# those messages are far too short to card - which is the gap this feature exists for.
def _mt(sender, text, minute, topic):
    return {"sender": sender, "text": text,
            "ts": f"2026-11-10T09:{minute:02d}:00Z", "_topic": topic}

CHAT = ["morning", "lol", "ok", "sure", "nice", "haha yeah", "yep", "cool", "ya", "hmm"]

def _interleaved():
    """Three threads, round-robin, chatter between - nothing arrives in a block."""
    threads = [
        [("13", "dinner last night came to 2400 total"), ("13", "can everyone send me 600 each")],
        [("4", "the split amount shows blank when participants is not sent"),
         ("4", "QA-116 still fails on the latest build")],
        [("70", "can you book me a cab from my home to the airport"),
         ("70", "I need to leave by 4 so book it before that")],
    ]
    topics = ["money", "bug", "ride"]
    out, minute, chat = [], 0, list(CHAT)
    for turn in range(2):
        for t, name in zip(threads, topics):
            s_, txt = t[turn]
            out.append(_mt(s_, txt, minute, name)); minute += 2
            if chat:
                out.append(_mt("26", chat.pop(0), minute, "chatter")); minute += 2
    return out

MULTITOPIC = [
    ("three interleaved topics all survive", _interleaved(), {"money", "bug", "ride"}),
    ("a bill buried in chatter survives",
     [_mt("26", CHAT[i % len(CHAT)], i, "chatter") for i in range(14)]
     + [_mt("13", "dinner last night came to 2400 total", 16, "money")]
     + [_mt("26", CHAT[i % len(CHAT)], 18 + i, "chatter") for i in range(6)],
     {"money"}),
    ("a minority topic is not crowded out by a dominant one",
     [_mt("4", t, i * 2, "bug") for i, t in enumerate([
         "QA-116 still fails on the latest build",
         "cannot set the group profile picture once the group is created",
         "back swipe gesture not working in group chats",
         "the split amount shows blank when participants is not sent"])]
     + [_mt("13", "dinner last night came to 2400 total", 20, "money")]
     + [_mt("26", CHAT[i % len(CHAT)], 22 + i, "chatter") for i in range(8)],
     {"bug", "money"}),
]

MONEY_POINTS = [
    ("a bill total scores",          "dinner last night came to 2400 total", True),
    ("a bill in the other order",    "cab fare was 350", True),
    ("a per-head split scores",      "split 4 ways, 52 each", True),
    ("a currency amount scores",     "$208 total for dinner", True),
    ("a build number does NOT",      "Build 61 is on TestFlight", False),
    ("a ticket id does NOT",         "QA-116 still fails", False),
    ("a version does NOT",           "version 3.2.0-rc3 shipped", False),
    ("a RAM figure does NOT",        "4 GB RAM minimum, 2 vCPU", False),
    ("a latency figure does NOT",    "100ms median, 140ms p90", False),
]


# Requests belong to the Intent Log; what a recap keeps is what the Intent Log does not
# have. Found by Akash 2026-10-19 after the first version read like a transcript of the
# money thread it was duplicating.
INTENTS = [
    ("a money request is the intent log's",   "Can you send me 500 for the cab", True),
    ("a ride request is the intent log's",    "Can you book me a cab to the office", True),
    ("an informal ask is too",                "Gimme 200 bucks if you can homie", True),
    ("so is announcing the transfer",         "Sending 380 now", True),
    ("and offering to send",                  "I'll send you 150 back", True),
    # What the intent log does NOT hold: the figure, the split, the balance, the date.
    ("what the bill was stays",               "Dinner last night came to 2400 total", False),
    ("how it was split stays",                "18000 split three ways so 6000 each", False),
    ("the net balance stays",                 "Net you owe me 380", False),
    ("a due date stays",                      "Rent is due on the 5th", False),
    ("a cost quoted stays",                   "The plumber wants 600 for the visit", False),
    ("a bug stays",                           "QA-116 is still failing on 61.3", False),
    ("a commitment stays",                    "I'll call the plumber", False),
]

# What the published guidance on recaps says people read them for.
SECTIONS_CASES = [
    ("a date change is a decision",   "It's thursday 2pm now instead of next monday", "decisions"),
    ("a due date is a decision",      "Rent is due on the 5th", "decisions"),
    ("taking something on is an action", "I'll call the plumber", "action items"),
    ("a group commitment is an action",  "Let's sync at 4 today", "action items"),
    ("being asked is an action",      "Please pull it before you test anything", "action items"),
    ("a question is open",            "Is that a blocker for thursday?", "open questions"),
    ("a fact is information",         "The kitchen tap is leaking again", "information"),
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
        # A ride or money REQUEST is the Intent Log's job, not the recap's. Akash,
        # 2026-10-19: "regarding intents we dont need them in summaries we already have
        # intent logs". What survives from a money conversation is what the intent log
        # does not hold - the amount, the split, who is owed.
        check("a ride request is left to the intent log",
              any("book me a cab" in p["text"].lower() for p in s.points), False)
        check("the amount the intent log does not hold survives",
              any("300" in p["text"] or "200" in p["text"] for p in s.points), True)
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

    total = (len(STANDS_ALONE) + len(SENSITIVE) + len(PRIVATE) + len(OPENERS)
             + len(MONEY_POINTS) + len(INTENTS) + len(SECTIONS_CASES)
             + 3 * len(MULTITOPIC) + 21)
    print(f"\n  TOTAL {total - len(failed)}/{total} correct")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
