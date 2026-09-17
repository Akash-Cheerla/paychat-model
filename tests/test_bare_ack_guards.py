"""The two bare-reply guards in app.py (2026-09-17), and the cases they must NOT block.

A bare "ok"/"sure" is quiet when it has nothing to agree to:
  ruling 6   a third person's bare ok after someone clearly took a group request
  s1a        a bare yes after a shortfall that asks for nothing

Everything else a bare reply does must survive: a second bare answer after a bare answer
(ruling 7), clear words from a third person, an unprompted offer confirmed, a shortfall
that asks, a shortfall after a real request. The LAST message of each conversation is scored.

    python tests/test_bare_ack_guards.py --url http://127.0.0.1:8971/detect
"""
import argparse, time

import requests

MR = ("money", "ride")
# (name, room kind, [(sender, text)], want on the last message, why)
CASES = [
    ("g09: clear take, 3rd person okay", "group",
     [("A", "can someone book me a cab to HSR"), ("B", "let me book a cab"), ("C", "okay")], None, "ruling 6"),
    ("clear take (money), 3rd person sure", "group",
     [("A", "can someone send me 40 for lunch"), ("B", "i'll send it right now"), ("C", "sure")], None, "ruling 6"),
    ("clear take, 3rd person ok with emoji", "group",
     [("A", "can someone book me an uber to the airport"), ("B", "booking it now"), ("C", "ok 👍")], None, "ruling 6"),
    ("clear take, a 'lol' between, 3rd person ok", "group",
     [("A", "can someone send me 40 for lunch"), ("B", "sending the 40 now"), ("D", "lol"), ("C", "ok")], None, "ruling 6"),
    ("clear take, requester 'ok thanks', 3rd person ok", "group",
     [("A", "can someone book me a cab to HSR"), ("B", "let me book a cab"), ("A", "ok thanks"), ("C", "okay")], None, "ruling 6"),
    ("bare sure, a 'lol' between, bare sure", "group",
     [("A", "can someone send me 40 for lunch"), ("B", "sure"), ("D", "lol"), ("C", "sure")], "money", "ruling 7"),
    ("the clear take itself fires", "group",
     [("A", "can someone book me a cab to HSR"), ("B", "let me book a cab")], "ride", "s3c"),
    ("bare sure, then bare sure", "group",
     [("A", "can someone send me 40 for lunch"), ("B", "sure"), ("C", "sure")], "money", "ruling 7"),
    ("clear take, 3rd person clear words", "group",
     [("A", "can someone book me a cab to HSR"), ("B", "let me book a cab"), ("C", "i can book one too")], "ride", "ruling 6"),
    ("unprompted offer, then ok", "dm",
     [("A", "let me book a cab for you"), ("B", "ok")], "ride", "s3c"),
    ("shortfall, then sure", "dm",
     [("A", "im short 300 this month"), ("B", "sure")], None, "s1a"),
    ("shortfall, then okay", "dm",
     [("A", "money's really tight right now"), ("B", "okay")], None, "s1a"),
    ("shortfall that asks, then sure", "dm",
     [("A", "im short 300 this month, can you send it?"), ("B", "sure")], "money", "s2: a request"),
    ("request, then shortfall, then sure", "dm",
     [("A", "can you send me 300"), ("A", "im short this month"), ("B", "sure")], "money", "s2: a request"),
    ("shortfall, then a commitment", "dm",
     [("A", "im short 300 this month"), ("B", "im sorry, sending it now")], "money", "s1a"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8900/detect")
    a = ap.parse_args()
    tag = int(time.time())
    bad = 0
    s = requests.Session()
    print(f"  {'case':40} {'want':6} {'got':12} {'money':>6} {'ride':>6}  why")
    for n, (name, kind, msgs, want, why) in enumerate(CASES):
        room = f"{'group' if kind == 'group' else 'dm'}_bag_{tag}_{n}"
        d = {}
        for i, (sender, text) in enumerate(msgs):
            try:
                d = s.post(a.url, timeout=90, json={"text": text, "room_id": room, "sender": sender,
                                                    "message_id": f"{room}_{i}"}).json()
            except Exception:
                d = {}
        got = [x for x in (d.get("intents") or []) if x in MR]
        sc = (d.get("conversation_state") or {}).get("scores", {}) or {}
        ok = got == ([want] if want else [])
        bad += not ok
        print(f"  {name:40} {str(want):6} {str(got or None):12} {float(sc.get('money', 0)):6.3f} "
              f"{float(sc.get('ride', 0)):6.3f}  {why}{'' if ok else '   <-- FAIL'}")
    print(f"\n  {len(CASES) - bad}/{len(CASES)} correct")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
