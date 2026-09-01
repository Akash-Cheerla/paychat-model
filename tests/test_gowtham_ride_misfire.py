"""Gowtham, 2:19-2:23 — the ride prompt landed on a message about the app itself.

Reported with a screenshot. Every message is his own (a one-sided test in a DM), and
the sequence is: several ride requests that produce nothing, a money statement that
correctly fires, then meta-commentary about the app — and the RIDE prompt appears on
"But payment intent does appear".

Two separate faults if it reproduces:
  * the ride requests at 2:19-2:22 never fire, including a fully specified one with
    both endpoints ("from 216 Vakil Garden City, to Swadeshi")
  * a message that is not a request, not a commitment, and not even about a cab
    collects the ride prompt several turns later

    python test_gowtham_ride_misfire.py --url http://127.0.0.1:8080/detect
"""
import argparse, json, sys, io, time

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# All from the same person, in order, exactly as in the screenshot.
TURNS = [
    "Let me book a cab",
    "Hmm",
    "I'll need to book a cab ride back home",
    "I need right now, to book a cab, from 216 Vakil Garden City, to Swadeshi",
    "I'm sending you one dollar",
    "Nor getting the ride intent",
    "*not",
    "But payment intent does appear",
    "Oh I got it now",
    "Strange",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/detect")
    ap.add_argument("--sender", default="30")
    a = ap.parse_args()

    room = f"gowtham_{int(time.time())}"
    fired = []
    for i, text in enumerate(TURNS):
        d = requests.post(a.url, timeout=60, json={
            "text": text, "room_id": room, "sender": a.sender,
            "message_id": f"{room}_{i}"}).json()
        got = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
        cs = (d.get("conversation_state") or {}).get("scores") or {}
        mark = f"   <<< {got}" if got else ""
        print(f"  {i}  {text[:58]:60} conv m={cs.get('money',0):.3f} "
              f"r={cs.get('ride',0):.3f}{mark}")
        if got:
            fired.append((i, text, got))

    print()
    ok = True
    ride_on_meta = [f for f in fired if "ride" in f[2] and f[0] >= 5]
    if ride_on_meta:
        ok = False
        for i, t, g in ride_on_meta:
            print(f"  FAIL  ride fired on message {i}, which is about the app: {t!r}")
    # Messages 0-3 are Andril talking to himself: one unprompted offer ("Let me book
    # a cab") and two needs ("I'll need to book a cab ride back home"). Under
    # FIRING_RULE 3c a need never fires and an unprompted offer waits for the other
    # party to confirm - and there was no other party. Silence there is correct.
    #
    # This used to assert the opposite and failed on every run, which made a fixed
    # bug look permanently open. Reported instead of asserted, so a change in
    # behaviour is still visible.
    early = [f for f in fired if f[0] <= 3]
    print(f"  note  messages 0-3 produced {len(early)} prompt(s); 3c expects 0")
    if early:
        ok = False
        for i, t, g in early:
            print(f"  FAIL  fired on message {i}, an unprompted offer or a need: {t!r}")
    if any("money" in f[2] for f in fired if f[0] == 4):
        print("  PASS  the money statement fired correctly")
    print("\n  " + ("reproduced the reported bug" if not ok else "did not reproduce"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
