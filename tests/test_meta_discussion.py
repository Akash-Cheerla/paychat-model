"""Gowtham's false fire, 2026-08-20 23:05.

They were discussing the FIRING RULES over WhatsApp. Akash typed an example containing
"B: can you book me a cab to airport", Gowtham replied "Yes" to a question about the
time window, and a ride prompt appeared.

The window contains a request-shaped string and a textbook acceptance. Nothing in it is
a real booking. This is the second report of the same class — Andril's was "Oh I got it
now" while describing the app — so it is a category, not an incident: talking ABOUT the
product operates the product.

The self-acknowledgement guard cannot catch this one. The quoted request came from the
OTHER speaker, so has_open() is legitimately true and "Yes" is legitimately an answer.
"""
import argparse, sys, io, time
import requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# G = Gowtham (the phone the screenshot came from), A = the other party.
TURNS = [
    ("G", "May be forever for now"),
    ("A", "Let's say\nB: can you book me a cab to airport (around 9 AM)\nA: okay(6-7PM)"),
    ("A", "So no limit and as long as it's in the window"),
    ("G", "Yes"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8150/detect")
    a = ap.parse_args()
    tag = int(time.time())
    room = f"meta_{tag}"
    bad = False
    for ti, (spk, txt) in enumerate(TURNS):
        d = requests.post(a.url, timeout=60, json={
            "text": txt, "room_id": room, "sender": spk,
            "message_id": f"{tag}_{ti}"}).json()
        got = [i for i in (d.get("intents") or []) if i in ("money", "ride")]
        sc = (d.get("conversation_state") or {}).get("scores", {})
        flat = txt.replace("\n", " / ")
        mark = f"   <<< FIRES {got}" if got else ""
        if got:
            bad = True
        print(f"  {spk}: {flat[:66]:66} ride={sc.get('ride',0):.3f} money={sc.get('money',0):.3f}{mark}")
    print(f"\n  {'REPRODUCED — prompt on a conversation about the product' if bad else 'no prompt — correct'}")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
