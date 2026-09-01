"""RECONSTRUCTION, not the real transcript.

The screenshot shows 4 messages; the classifier reads 10. This builds the kind of window
a rules discussion actually produces — several quoted example conversations, each one
carrying a genuine request-shaped string — and ends on the same "Yes".

If this fires, the mechanism is confirmed even though the exact scrollback is unknown:
quoted examples accumulate in the window and the model cannot tell a quotation from a
booking.
"""
import argparse, sys, io, time
import requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TURNS = [
    ("A", "quick one on the ride rules"),
    ("A", "Let's say\nB: can you book me a cab to the airport\nA: let me book a cab"),
    ("G", "Request and answer will always trigger"),
    ("A", "and if A says ok let me book a cab"),
    ("G", "Same thing, it fires"),
    ("G", "May be forever for now"),
    ("A", "Let's say\nB: can you book me a cab to airport (around 9 AM)\nA: okay(6-7PM)"),
    ("A", "So no limit and as long as it's in the window"),
    ("G", "Yes"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8151/detect")
    a = ap.parse_args()
    tag = int(time.time()); room = f"metaL_{tag}"; bad = False
    for ti, (spk, txt) in enumerate(TURNS):
        d = requests.post(a.url, timeout=60, json={
            "text": txt, "room_id": room, "sender": spk, "message_id": f"{tag}_{ti}"}).json()
        got = [i for i in (d.get("intents") or []) if i in ("money", "ride")]
        sc = (d.get("conversation_state") or {}).get("scores", {})
        flat = txt.replace("\n", " / ")
        if got: bad = True
        print(f"  {spk}: {flat[:62]:62} ride={sc.get('ride',0):.3f}"
              f"{'   <<< FIRES ' + str(got) if got else ''}")
    print(f"\n  {'REPRODUCED the mechanism' if bad else 'still no prompt'}")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
