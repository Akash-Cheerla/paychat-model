"""Ride slot controls — written BEFORE the fix so they fail first.

Two defects found replaying a real booking chat (+919945388748, 2026-08-10):

  STALE INHERITANCE. "Book a can from jp nagar to Banashankari" produced a prompt for
  "Bangalore -> Bengaluru" — the trip from two requests earlier. When extraction finds
  nothing on a fresh request, the merge falls back to whatever the previous fire
  resolved to, so instead of an empty field the user is shown a plausible, completely
  wrong trip and can tap straight through it. Blank is the correct failure.

  LINE BREAKS. An address typed on its own line loses the street:
      "book a cab for me\\nVakil garden city, Kanakapura road Bangalore\\nTo BLR airport"
          -> pickup "Bangalore", destination "Bengaluru"
      same text inline after "from"
          -> pickup "Vakil garden city, Kanakapura road Bangalore"  (correct)
  People send addresses on separate lines; the parser only handles one line.

Run:  python test_ride_slots.py [--url http://127.0.0.1:8005/detect]
"""
import argparse, itertools, sys, time

import requests

# Seeded from the clock, NOT a constant. Room state lives on the server for
# CONV_CONTEXT_TIME_CAP (4h), so a fixed start meant every re-run replayed into rooms
# that already held fires from the previous run — recently_fired then suppressed them
# and two green controls turned red with no code change. Tests must not share rooms
# with their own history.
_ids = itertools.count(int(time.time() * 1000) % 1_000_000_000)


def play(url, convo):
    """Replay a conversation, return (pickup, destination) of each prompt it fires."""
    n = next(_ids)
    room = f"dm_{n}_{n + 40}"
    fired = []
    for i, (who, text) in enumerate(convo):
        r = requests.post(url, json={"text": text, "room_id": room, "sender": who,
                                     "message_id": f"m{i}"}, timeout=180).json()
        cs = r.get("conversation_state") or {}
        if [x for x in (r.get("intents") or []) if x in ("money", "ride")]:
            s = (cs.get("triggered_by") or {}).get("slots") or {}
            fired.append((s.get("pickup"), s.get("destination")))
    return fired


def norm(v):
    return (v or "").lower().replace(",", " ").split()


CASES = [
    # ── stale inheritance ──
    ("stale: unparseable new request must NOT inherit the last trip",
     [("A", "book a cab from jp nagar to banashankari"), ("B", "Sure"),
      ("A", "book smth for me"), ("B", "Sure")],
     lambda f: len(f) < 2 or f[1] == (None, None),
     "second prompt blank, not jp nagar -> banashankari"),

    ("stale: a NEW request with its own places wins",
     [("A", "book a cab from jp nagar to banashankari"), ("B", "Sure"),
      ("A", "book a cab from indiranagar to whitefield"), ("B", "Sure")],
     lambda f: len(f) == 2 and "indiranagar" in norm(f[1][0]) and "whitefield" in norm(f[1][1]),
     "second prompt indiranagar -> whitefield"),

    ("continuation of the SAME trip keeps its slots",
     [("A", "book me a cab to koramangala"), ("A", "im at indiranagar"), ("B", "Sure")],
     lambda f: len(f) == 1 and "indiranagar" in norm(f[0][0]) and "koramangala" in norm(f[0][1]),
     "one prompt, indiranagar -> koramangala"),

    # ── line breaks ──
    ("multi-line address keeps the street",
     [("A", "Can you book a cab for me\nVakil garden city, Kanakapura road Bangalore\n"
            "To Bengaluru international airport terminal 2"), ("B", "Sure")],
     lambda f: len(f) == 1 and "vakil" in norm(f[0][0]) and "airport" in norm(f[0][1]),
     "pickup keeps 'vakil garden city', destination keeps 'airport'"),

    ("multi-line matches the inline wording",
     [("A", "Can you book a cab for me from Vakil garden city, Kanakapura road Bangalore "
            "to Bengaluru international airport terminal 2"), ("B", "Sure")],
     lambda f: len(f) == 1 and "vakil" in norm(f[0][0]) and "airport" in norm(f[0][1]),
     "same as the multi-line version"),

    ("airport name is not truncated to the city",
     [("A", "book me a cab to bengaluru international airport terminal 2"), ("B", "Sure")],
     lambda f: len(f) == 1 and "airport" in norm(f[0][1]),
     "destination contains 'airport', not just 'bengaluru'"),

    # ── must not regress ──
    ("plain from/to still works",
     [("A", "book a cab from jp nagar to Banashankari"), ("B", "Sure")],
     lambda f: len(f) == 1 and "jp" in norm(f[0][0]) and "banashankari" in norm(f[0][1]),
     "jp nagar -> banashankari"),

    ("'my location' is kept as written",
     [("A", "book a cab from my location to terminal 2"), ("B", "Sure")],
     lambda f: len(f) == 1 and "location" in norm(f[0][0]),
     "pickup 'My Location', for the user to edit"),

    ("address stated separately fills an empty pickup",
     [("A", "book me a cab to the airport"),
      ("A", "im at vakil garden city, kanakapura road"), ("B", "Sure")],
     lambda f: len(f) == 1 and "vakil" in norm(f[0][0]),
     "pickup recovered from the address message"),

    ("rider changes their mind mid-request",
     [("A", "book me a cab to koramangala"), ("A", "actually make it indiranagar"),
      ("B", "on it")],
     lambda f: len(f) == 1 and "indiranagar" in norm(f[0][1]),
     "destination indiranagar"),

    # ── one request, several bookers (group) ──
    #
    # We are a surface, not a dispatcher. Anyone who says they will book gets a sheet
    # with the destination filled; whoever actually taps it wins. Deciding who the
    # "real" booker is would mean modelling withdrawal and headcount, and the second
    # booker was getting a BLANK destination because the first fire consumed the
    # request record.
    ("two bookers on one request both get the destination",
     [("A", "we're 7 people, need 2 cabs to the airport"),
      ("B", "i'll book one"), ("C", "ok i'll book the other one")],
     lambda f: len(f) == 2 and all("airport" in norm(d) for _, d in f),
     "both prompts carry 'airport'"),

    # Two claims, kept apart on purpose. The donation logic has no cap — whoever fires
    # gets the destination. Whether a third volunteer fires AT ALL is the model's call,
    # and today it does not (see the KNOWN-RED case below).
    ("a later booker still gets the destination",
     [("A", "can someone book a cab to the airport"),
      ("B", "i'll book it"), ("C", "i'll book it"), ("D", "i'll book it")],
     lambda f: len(f) >= 2 and all("airport" in norm(d) for _, d in f),
     "every prompt that fires carries 'airport'"),

    ("the same person committing twice does not re-take the request",
     [("A", "can someone book a cab to the airport"),
      ("B", "i'll book it"), ("B", "booking it now")],
     lambda f: len(f) == 1,
     "one prompt for B, the restatement is an echo"),

    # ── guards: a divisible request must not outlive its own trip ──
    ("a NEW trip does not inherit the divisible destination",
     [("A", "we're 7 people, need 2 cabs to the airport"), ("B", "i'll book one"),
      ("A", "also book a cab from indiranagar to whitefield"), ("C", "i'll book it")],
     lambda f: len(f) == 2 and "whitefield" in norm(f[1][1]),
     "second prompt whitefield, NOT airport"),

    # Deliberately the OPPOSITE of the DM case at the top of this file, and the
    # difference is who is speaking. There, the same person re-commits and take()
    # refuses a second helping, so the sheet is blank. Here a DIFFERENT person
    # volunteers, and the airport trip is still the only trip anyone has described —
    # "book smth for me" scores 0.066 and is not a ride request at all, so it does not
    # replace anything. Pre-filling Airport is the call (2026-08-10): we surface the
    # intent someone expressed and let them edit or ignore it, rather than handing them
    # an empty sheet to retype.
    ("a different volunteer after a vague ask keeps the live trip",
     [("A", "we're 7 people, need 2 cabs to the airport"), ("B", "i'll book one"),
      ("A", "book smth for me"), ("C", "i'll book it")],
     lambda f: len(f) == 2 and "airport" in norm(f[1][1]),
     "second prompt keeps airport"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8005/detect")
    a = ap.parse_args()
    bad = 0
    for name, convo, check, expect in CASES:
        got = play(a.url, convo)
        ok = check(got)
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"         got    {got}")
            print(f"         expect {expect}")
    print(f"\n  {len(CASES) - bad}/{len(CASES)} passing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
