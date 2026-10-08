"""A split shows each person their SHARE, not the whole bill.

Found live on 2026-10-07. The team sent this into a group and every member's sheet opened
at $200:

    Akash:   Can everyone send me $200 for the dinner yesterday?
    Gowtham: Was that the total or for each one Akash?
    Akash:   Total , please pay up individual shares
    Gowtham: Okay. Sending my share.          -> sheet said $200, should say $25

Dividing needs two independent things to be true, and that message got only one of them:
_DIVISIBLE (is this a split at all) and _TOTAL_AMT (is the figure the whole bill). Three
separate holes came out of it:

  1. "everyone send me $200" did not count as a total, so $200 was read as what EACH
     person owes. Asking a group to each send the same stated figure is not how people
     split a bill - the figure IS the bill (_TOTAL_BY_GROUP).
  2. "split 8 ways" was ignored. The headcount is in the message and _WAYS already reads
     it, but dividing was refused because the word "total" was absent.
  3. "pay up individual shares" was not a split: the pattern wanted a pronoun directly
     before "shares", so the adjective broke it - which is why the clarification above was
     filed as an ordinary request and carried the full bill.

The safe failure matters as much as the fix: with no `participants` the amount is left
BLANK rather than showing the total. Pre-filling 200 for someone who owes 25 is the one
direction a payment prompt must never be wrong in.

    python tests/test_split_share.py --url http://127.0.0.1:8971/detect
"""
import argparse
import io
import sys
import time

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# (name, participants, [(sender, text)], expected amount on each fire)
CASES = [
    ("the live 2026-10-07 sequence, room of 8", 8,
     [("20", "Can everyone send me $200 for the dinner yesterday?"),
      ("10", "Was that the total or for each one Akash?"),
      ("20", "Total , please pay up individual shares"),
      ("10", "Okay. Sending my share."),
      ("53", "Okay, sending my portion")],
     "$25"),

    ("the same bill in a room of 4", 4,
     [("20", "Can everyone send me $200 for the dinner yesterday?"),
      ("10", "Okay. Sending my share.")],
     "$50"),

    # Without a headcount we cannot know the share, so the field is left empty. This is
    # the whole point: blank is an annoyance, the full total is a wrong payment.
    ("no participants sent: blank, never the total", None,
     [("20", "Can everyone send me $200 for the dinner yesterday?"),
      ("10", "Okay. Sending my share.")],
     None),

    ("a stated headcount divides without the word total", 8,
     [("20", "Can everyone send me $200 for dinner, split 8 ways"),
      ("10", "Sending my share")],
     "$25"),

    ("individual shares, no pronoun", 8,
     [("20", "Dinner was $200, pay up individual shares"),
      ("10", "Okay sending")],
     "$25"),

    ("the wordings that already worked still work", 8,
     [("20", "Dinner yesterday was $200 total, can everyone send me their share?"),
      ("10", "Sending my share")],
     "$25"),

    ("split between the N of us", 8,
     [("20", "Dinner was $200, split between the 8 of us"),
      ("10", "Sending now")],
     "$25"),

    # A figure already stated per-person must NOT be divided again - that gave $25 -> $3.
    ("a per-person figure is never divided again", 8,
     [("20", "Can everyone send me $25 each for the dinner yesterday?"),
      ("10", "Okay. Sending my share.")],
     "$25"),

    # A bare number comes out of slot extraction marked "$", so the share keeps that mark.
    ("the trip case from the eval set", 5,
     [("20", "the trip came to 5000, send me your shares"),
      ("10", "sending")],
     "$1000"),

    # One-to-one asks are not splits and must be untouched.
    ("a DM-style one-to-one ask keeps its figure", 8,
     [("20", "can you send me $200 for dinner"),
      ("10", "Sure")],
     "$200"),

    ("a debt is not a split", 8,
     [("20", "you owe me $200"),
      ("10", "sure sending")],
     "$200"),

    # Uneven splits USED to blank, which in a real group is the normal case and not an
    # edge one: group_12 has 9 members, so only multiples of 9 divided and the live $208
    # and $200 bills both came back empty. Ruled by Akash 2026-10-07 - round it, the way
    # every split app does. The asker is a cent short and the field is editable.
    ("an uneven split rounds to the cent", 3,
     [("20", "Can everyone send me $100 for dinner"),
      ("10", "Sending my share")],
     "$33.33"),

    # The live room: 9 members, $208. This is the bill that came back blank.
    ("the live 9-person $208 bill", 9,
     [("20", "Dinner yesterday was $208 total, can everyone send me their share?"),
      ("10", "sending")],
     "$23.11"),

    # Paise are not a thing anyone splits, and "Rs 23.11" reads as a bug.
    # Slot extraction normalises "Rs" to the symbol, so the share carries the symbol too.
    ("rupees round to the whole rupee", 7,
     [("20", "Dinner was Rs 1000 total, send me your shares"),
      ("10", "sending")],
     "₹143"),

    # A word-form currency sits AFTER the number, and substituting a stripped prefix put
    # it in front: "50 rupees" split 3 ways returned "rupees17".
    ("a word-form currency keeps its side", 3,
     [("20", "Dinner was 50 rupees total, send me your shares"),
      ("10", "sending")],
     "17 rupees"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/detect")
    a = ap.parse_args()
    tag = int(time.time())
    s = requests.Session()
    failed = []
    for ci, (name, parts, seq, want) in enumerate(CASES):
        room = f"group_sh{tag}_{ci}"
        got, fired_any = [], False
        for j, (sender, text) in enumerate(seq):
            body = {"text": text, "room_id": room, "sender": sender,
                    "message_id": f"{tag}_{ci}_{j}"}
            if parts:
                body["participants"] = parts
            d = s.post(a.url, json=body, timeout=90).json()
            if [x for x in (d.get("intents") or []) if x in ("money", "ride")]:
                fired_any = True
                got.append((d.get("slots") or {}).get("amount"))
        ok = fired_any and all(g == want for g in got)
        if not ok:
            failed.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"        want {want!r} on every prompt, got {got!r}"
                  + ("  (nothing fired)" if not fired_any else ""))
    print(f"\n  {len(CASES) - len(failed)}/{len(CASES)} passing")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
