"""The four false fires the team actually hit, 2026-08-12 to 2026-08-21.

Taken from the dogfood log, which is real team traffic and stays gitignored - the
sequences are reproduced here in the shortest form that still triggers them, with names
removed, because a regression test that lives only in an ungitted log is not a test.

Three of the four are the same category: talking about the product operates the product.
The team is currently the whole user base, so this is not a corner case for us.

  A  "But payment intent does appear"   ride 0.931  describing the app
  B  "Yes" after "I think that's the bug"  money 0.996  debugging the app
  C  "Yes" in a discussion OF the firing rules   ride  quoting an example
  D  "Hi"                                  money 0.997  a plain greeting

Only A can be caught by the self-acknowledgement guard - it is one speaker talking to
himself. In B, C and D the open request genuinely came from the other person, so the
guard is right not to intervene and something else has to.
"""
import argparse, sys, io, time
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CASES = [
    ("A  describing the app", "ride", [
        ("4", "Let me book a cab"),
        ("4", "Hmm"),
        ("4", "I'll need to book a cab ride back home"),
        ("4", "I need right now, to book a cab, from 216 Vakil Garden City"),
        ("4", "I'm sending you one dollar"),
        ("4", "Nor getting the ride intent"),
        ("4", "*not"),
        ("4", "But payment intent does appear"),
    ]),
    ("B  debugging the app", "money", [
        ("26", "Oh"),
        ("26", "But why"),
        ("11", "I think that's the bug"),
        ("26", "Ohk"),
        ("11", "It's ok"),
        ("11", "I will try to send money"),
        ("11", "Hi"),
        ("26", "Yes"),
    ]),
    ("C  discussing the firing rules", "ride", [
        ("26", "The intent is not displaying."),
        ("20", "How long shall we keep the unanswered request open?"),
        ("10", "May be forever for now"),
        ("20", "Let's say\nB: can you book me a cab to airport (around 9 AM)\nA: okay(6-7PM)"),
        ("20", "So no limit and as long as it's in the window"),
        ("10", "Yes"),
    ]),
    ("D  a plain greeting", "money", [
        ("53", "How are you"),
        ("53", "Hi"),
        ("26", "Hi"),
        ("26", "How are you"),
        ("26", "Hi"),
        ("26", "Can you send me 10$"),
        ("53", "Hi"),
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8400/detect")
    a = ap.parse_args()
    tag = int(time.time())
    bad = 0
    for ci, (name, intent, turns) in enumerate(CASES):
        room = f"df_{tag}_{ci}"
        last = None
        for ti, (spk, txt) in enumerate(turns):
            d = requests.post(a.url, timeout=60, json={
                "text": txt, "room_id": room, "sender": spk,
                "message_id": f"{tag}_{ci}_{ti}"}).json()
            got = [i for i in (d.get("intents") or []) if i in ("money", "ride")]
            sc = (d.get("conversation_state") or {}).get("scores", {}) or {}
            last = (got, sc.get(intent, 0.0), txt)
        got, score, txt = last
        fired = intent in got
        bad += fired
        flat = txt.replace("\n", " / ")
        print(f"  {name:32} {intent}={score:.3f}  "
              f"{'STILL FIRES' if fired else 'quiet'}   {flat[:40]!r}")
    print(f"\n  {len(CASES)-bad}/{len(CASES)} now correct")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
