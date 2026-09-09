"""A location belonging to someone else in the chat.

Reported by Andril on 2026-09-08, in the Wisecats group:

    "can anyone book a ride from my current location to Brahma's current location"
    "It can't differentiate and identify names and aliases for users, so this doesn't work"

The screenshot showed the sheet opening with pickup "216, Vakil Garden City, Bengaluru"
and drop rendered as the literal string "Brahma's location". The pickup resolved because
"my current location" is in _GPS_PLACES; the drop did not, because nothing recognised a
person's name as a place, so it fell through as ordinary text.

The server does NOT resolve the name and should not. /detect receives `participants` as an
integer headcount with no roster, and the whole location design exists so that
whose-location-is-this leaves as a hint rather than arriving as a coordinate. So the name
is reported and whoever holds the roster matches it. A name matching nobody falls back to
`ask`, which is what makes it safe to be permissive here - "the driver's location" and
"mom's house" cost nothing.

Two rulings from Gowtham the same night are encoded:

    "It should crack it for a full matching first name that's part of the conversation"
    "location should always make a request for current location, unless it says home
     location or residence or the like"

So place=None means ask that person where they are now; only home/office wordings point
at a saved place.

    python tests/test_participant_location.py [--url http://127.0.0.1:8970/detect]
"""
import argparse, io, sys, time

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# (name, turns, slot, expected resolve, expected name, expected place)
CASES = [
    ("Andril's report, verbatim",
     [("A", "Can anyone book me a cab from my current location to Brahma's location?"),
      ("B", "Sure , let me book it")],
     "destination", "participant", "Brahma", None),

    ("the pickup in the same message still resolves to GPS",
     [("A", "Can anyone book me a cab from my current location to Brahma's location?"),
      ("B", "Sure , let me book it")],
     "pickup", "gps", None, None),

    ("a person's house is their SAVED home, not their position",
     [("A", "book me a cab from my location to Samyak's house"), ("B", "sure")],
     "destination", "participant", "Samyak", "home"),

    # This read as a broken extractor for an hour. It was a broken TEST: the third turn
    # was "thanks", so the fire happened on turn 2 and the assertion was reading turn 3's
    # response, which correctly carries no slots. The conversation now ends on the message
    # that fires. Kept as a note because "the slots are null" and "the last message did not
    # fire" look identical from outside and this will happen again.
    ("a person's office is their saved office",
     [("A", "get me a cab to Andril's office"), ("B", "yeah booking")],
     "destination", "participant", "Andril", "office"),

    # The slot extractor title-cases before the hint sees the phrase, so the name comes
    # back capitalised regardless of how it was typed. The resolver matches case-
    # insensitively, so this is cosmetic - recorded because it surprised me once.
    ("possessive with no apostrophe still resolves the name",
     [("A", "book a cab to brahmas location"), ("B", "sure")],
     "destination", "participant", "Brahma", None),

    # --- must NOT be read as a participant ---
    ("a determiner is never a name",
     [("A", "book me a cab to the driver's location"), ("B", "sure")],
     "destination", None, None, None),

    ("an ordinary named place gets no hint at all",
     [("A", "book me a cab from koramangala to the airport"), ("B", "sure")],
     "destination", None, None, None),

    ("the speaker's own saved place is still saved_place",
     [("A", "book me a cab from home to the airport"), ("B", "sure")],
     "pickup", "saved_place", None, "home"),
]


def field(hint, slot):
    for f in (hint or {}).get("fields") or []:
        if f.get("slot") == slot:
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8970/detect")
    a = ap.parse_args()
    tag = int(time.time())
    ok = 0
    for ci, (name, turns, slot, want_r, want_n, want_p) in enumerate(CASES):
        room = f"plt_{tag}_{ci}"
        d = {}
        for j, (s, t) in enumerate(turns):
            d = requests.post(a.url, timeout=90, json={
                "text": t, "room_id": room, "sender": s,
                "message_id": f"{tag}_{ci}_{j}", "participants": 4}).json()
        f = field(d.get("needs_location"), slot)
        got_r = f.get("resolve") if f else None
        got_n = f.get("name") if f else None
        got_p = f.get("place") if f else None
        good = (got_r == want_r and got_n == want_n and got_p == want_p)
        ok += good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")
        if not good:
            print(f"          slot {slot}: got resolve={got_r!r} name={got_n!r} "
                  f"place={got_p!r}")
            print(f"                  want resolve={want_r!r} name={want_n!r} "
                  f"place={want_p!r}")
    print(f"\n  {ok}/{len(CASES)} passing")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
