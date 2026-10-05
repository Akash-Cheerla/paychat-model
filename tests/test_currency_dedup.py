"""A different currency is a different request.

Found by adjudicating the 2026-09-30..10-03 dogfood window, where it swallowed two real
requests in four days:

    20: Hey can send me $50 for dinner        -> prompt
    20: Can you also send me 50 rupees ...    -> nothing at all
    53: I need 10₹ can you send me?           -> prompt
    53: Can you send me 10$                   -> nothing at all

_amount_key keeps the digits and drops the symbol, so the second request read as the first
one re-sent and ruling 2 suppressed the acceptance. A team working across INR and USD on a
product aimed at India hits this constantly.

The fix has to leave ruling 2 itself intact: the SAME request sent again still earns no
second prompt, and an amount that names no currency still counts as the same one restated.

    python tests/test_currency_dedup.py --url http://127.0.0.1:8971/detect
"""
import argparse
import io
import sys
import time

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RUPEE = "₹"
EURO = "€"

# (name, [(sender, text)], [expected intent per message])
CASES = [
    ("dollars then rupees, same number, from the log",
     [("20", "Hey can send me $50 for dinner"), ("67", "Sure"),
      ("20", "Can you also send me 50 rupees for the cab ?"), ("67", "Sure")],
     [None, "money", None, "money"]),

    ("rupees then dollars, same number, from the log",
     [("53", f"I need 10{RUPEE} can you send me?"), ("67", "Oh ok"),
      ("53", "Can you send me 10$"), ("67", "Sure")],
     [None, "money", None, "money"]),

    ("euros then dollars",
     [("20", f"can you send me {EURO}20"), ("67", "Sure"),
      ("20", "can you send me $20"), ("67", "Sure")],
     [None, "money", None, "money"]),

    # Ruling 2 (Akash, 2026-09-15) must survive: the same request re-sent right after it
    # was accepted gets no second prompt.
    ("same currency, same amount: one prompt only",
     [("20", "can you send me $50"), ("67", "Sure"),
      ("20", "can you send me $50"), ("67", "Sure")],
     [None, "money", None, None]),

    ("a bare number after a currency is the same request restated",
     [("20", "can you send me $50"), ("67", "Sure"),
      ("20", "can you send me 50"), ("67", "Sure")],
     [None, "money", None, None]),

    ("a currency after a bare number is the same request restated",
     [("20", "can you send me 50"), ("67", "Sure"),
      ("20", "can you send me $50"), ("67", "Sure")],
     [None, "money", None, None]),

    # Same currency written two ways is still one request.
    ("rs and the symbol are the same currency",
     [("20", "can you send me Rs 500"), ("67", "Sure"),
      ("20", f"can you send me {RUPEE}500"), ("67", "Sure")],
     [None, "money", None, None]),

    ("dollars and the word are the same currency",
     [("20", "can you send me $50"), ("67", "Sure"),
      ("20", "can you send me 50 dollars"), ("67", "Sure")],
     [None, "money", None, None]),

    ("control: a different amount always fires",
     [("20", "Hey can send me $50 for dinner"), ("67", "Sure"),
      ("20", "Can you send me $25 for the ride"), ("67", "Sure")],
     [None, "money", None, "money"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/detect")
    a = ap.parse_args()
    tag = int(time.time())
    s = requests.Session()
    failed = []
    for ci, (name, seq, want) in enumerate(CASES):
        room = f"dm_cur{tag}{ci}_{tag}{ci + 1}"
        got = []
        for j, (sender, text) in enumerate(seq):
            d = s.post(a.url, timeout=90, json={
                "text": text, "room_id": room, "sender": sender,
                "message_id": f"{tag}_{ci}_{j}"}).json()
            fired = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
            got.append(fired[0] if fired else None)
        ok = got == want
        if not ok:
            failed.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            for (sender, text), w, g in zip(seq, want, got):
                mark = " " if w == g else "<"
                print(f"        {mark} {sender}: {text[:52]:52} want={w} got={g}")
    print(f"\n  {len(CASES) - len(failed)}/{len(CASES)} passing")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
