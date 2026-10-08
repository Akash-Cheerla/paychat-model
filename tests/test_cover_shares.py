"""Paying for more than one person pays more than one share.

Ruled by Akash, 2026-10-07, on the $90 / 5-person dinner split: "ill cover me and priya"
is a commitment to TWO shares, $36, not one. Covering a member who is short is ordinary
in a group, and every one of these prompts opened at a single $18 - leaving the asker
short by exactly the share of the person being covered.

Counting is narrow on purpose, because an error here makes the payment too LARGE:
  - "both" and "the N of us" state the count outright.
  - a NAME counts only when it matches the roster the caller sent, so "my share and the
    tip" can never read as two people, and neither can a name we cannot verify.

KNOWN GAP, not covered here: the "I'll pay / I'll cover / I'll settle for X" family does
not FIRE at all. Measured 2026-10-07 on the conversation classifier, after a split request:

    sending my share            0.995  fires
    sending for both of us      0.995  fires
    sending mine and priyas     0.995  fires
    ill cover it                0.994  fires
    ill cover both of us        0.980  quiet   <- five thousandths under the bar
    ill take care of it         0.897  quiet
    ill cover priya too         0.691  quiet
    ill pay priya's share too   0.081  quiet
    ill cover me and priya      0.067  quiet
    ill pay for both of us      0.042  quiet
    ill settle for both         0.037  quiet
    let me cover priya's share  0.025  quiet

"sending ..." always fires; "I'll pay ..." mostly does not, and naming a second person
collapses the score ("ill cover it" 0.994 -> "ill cover me and priya" 0.067). That is a
training-distribution gap: "I'll pay for both of us" (a commitment) and "I'll pay you next
week" (a deferral that must stay quiet) are the same surface form, so no pattern separates
them. It belongs in the next training round, NOT in a rules-based fire. The cases below
therefore use the wordings that do fire, and the ruling's arithmetic is what is tested.

    python tests/test_cover_shares.py --url http://127.0.0.1:8971/detect
"""
import argparse
import io
import sys
import time

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROSTER = [
    {"id": "20", "name": "Akash"},
    {"id": "10", "name": "Gowtham"},
    {"id": "30", "name": "Priya"},
    {"id": "40", "name": "Raj"},
    {"id": "50", "name": "Sunny"},
]
ASK = "Dinner was $90 total, send me your shares"

# (name, send_roster, reply, expected amount)
CASES = [
    ("one share is still one share", True, "sending my share", "$18"),
    ("both of us pays two shares", True, "sending for both of us", "$36"),
    ("a roster name plus the speaker", True, "sending mine and priyas", "$36"),
    ("a stated headcount pays that many", True, "sending for the 3 of us", "$54"),

    # The negative controls matter more than the positives: every one of these would be
    # an OVERCHARGE if the counting were loose.
    ("a bare word after 'and' is not a person", True, "sending my share and the tip", "$18"),
    ("an unverifiable name is not counted", False, "sending mine and priyas", "$18"),
    ("a name nobody in the room has", True, "sending mine and deepaks", "$18"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/detect")
    a = ap.parse_args()
    tag = int(time.time())
    s = requests.Session()
    failed = []
    for ci, (name, with_roster, reply, want) in enumerate(CASES):
        room = f"group_cv{tag}_{ci}"
        base = {"room_id": room, "participants": 5}
        if with_roster:
            base["roster"] = ROSTER
        s.post(a.url, json={**base, "text": ASK, "sender": "20",
                            "message_id": f"{tag}_{ci}_0"}, timeout=90)
        d = s.post(a.url, json={**base, "text": reply, "sender": "10",
                                "message_id": f"{tag}_{ci}_1"}, timeout=90).json()
        fired = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
        got = (d.get("slots") or {}).get("amount") if fired else None
        ok = bool(fired) and got == want
        if not ok:
            failed.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"        {reply!r} -> want {want!r}, got {got!r}"
                  + ("  (nothing fired)" if not fired else ""))
    print(f"\n  {len(CASES) - len(failed)}/{len(CASES)} passing")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
