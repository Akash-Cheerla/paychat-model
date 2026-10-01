"""needs_location with a `roster`: "<Name>'s address" and "your place" resolve to a user.

Before 2026-10-01 the model reported the name (`resolve: "participant"`) and nobody
downstream matched it, so the sheet showed the literal text "Akash's address". Now the
caller sends who is in the room - first name, nickname, and the NAMES of the saved places
each person has - and the hint names the owner on every field (`fields[].user_id`).

Each case sends the request WITH a roster and checks the field for one slot. Rules under
test: exact first-name or nickname match, one match only (two Akash -> ask), a saved
place only when that person has it (else ask), "your" only in a DM, "address" = home,
curly apostrophes, and no roster = the old `participant` answer.

    python tests/test_roster_location.py [--url http://127.0.0.1:8000/classify]
"""
import argparse, io, sys, time

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

GROUP = [
    {"id": "10", "name": "Gowtham", "nickname": None, "places": ["home"]},
    {"id": "20", "name": "Akash", "nickname": "AK", "places": ["home", "office"]},
    {"id": "4", "name": "Andril", "nickname": None, "places": []},
]
TWO_AKASH = GROUP + [{"id": "5", "name": "Akash", "nickname": None, "places": []}]
DM = [
    {"id": "30", "name": "Rupesh", "nickname": None, "places": []},
    {"id": "31", "name": "Yash", "nickname": None, "places": ["home"]},
]

# (name, room, roster, turns, slot, expected field subset)
CASES = [
    ("Akash's address is Akash's saved home",
     "group", GROUP, [("10", "book a cab from my home to Akash's address")],
     "destination", {"resolve": "saved_place", "place": "home", "user_id": "20"}),

    ("the speaker's own home carries the speaker's id",
     "group", GROUP, [("10", "book a cab from my home to Akash's address")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "10"}),

    ("curly apostrophe, and still on the fire",
     "group", GROUP, [("10", "can someone book a cab from my location to Akash’s address"),
                      ("4", "I got you")],
     "destination", {"resolve": "saved_place", "place": "home", "user_id": "20"}),

    ("a nickname matches, current location is gps",
     "group", GROUP, [("10", "book a cab from my location to AK's current location")],
     "destination", {"resolve": "gps", "place": None, "user_id": "20"}),

    ("a saved place the person does not have is ask, owner known",
     "group", GROUP, [("10", "get me a cab to Andril's office")],
     "destination", {"resolve": "ask", "place": "office", "user_id": "4"}),

    ("a name nobody has is ask, name kept",
     "group", GROUP, [("10", "book me a cab from my location to Brahma's place")],
     "destination", {"resolve": "ask", "name": "Brahma", "user_id": None}),

    ("two members with the same name is ask, never a guess",
     "group", TWO_AKASH, [("10", "book a cab from my location to Akash's house")],
     "destination", {"resolve": "ask", "name": "Akash", "user_id": None}),

    ("the speaker without an office saved is asked at once",
     "group", GROUP, [("4", "book me a cab from my office to the airport")],
     "pickup", {"resolve": "ask", "place": "office", "user_id": "4"}),

    ("your place in a DM is the other party's home",
     "dm", DM, [("30", "i will book you a cab from your place to the airport")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "31"}),

    ("your location in a DM is the other party's position",
     "dm", DM, [("31", "let me book you a cab from your current location to the airport")],
     "pickup", {"resolve": "gps", "place": None, "user_id": "30"}),

    ("your location in a group is ask",
     "group", GROUP, [("10", "i will book you a cab from your location to the airport")],
     "pickup", {"resolve": "ask", "who": "other_party"}),

    ("no roster: the pre-roster answer, unchanged",
     "group", None, [("10", "book a cab from my location to Akash's address")],
     "destination", {"resolve": "participant", "name": "Akash", "place": "home"}),
]


def field(hint, slot):
    for f in (hint or {}).get("fields") or []:
        if f.get("slot") == slot:
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/classify")
    a = ap.parse_args()
    tag = int(time.time())
    ok = 0
    for ci, (name, kind, roster, turns, slot, want) in enumerate(CASES):
        room = f"dm_{tag}{ci}_{tag}{ci + 1}" if kind == "dm" else f"grp_{tag}_{ci}"
        d = {}
        for j, (s, t) in enumerate(turns):
            body = {"text": t, "room_id": room, "sender": s,
                    "message_id": f"{tag}_{ci}_{j}", "participants": len(roster or [])}
            if roster is not None:
                body["roster"] = roster
            d = requests.post(a.url, timeout=90, json=body).json()
        f = field(d.get("needs_location"), slot) or {}
        good = all(f.get(k) == v for k, v in want.items())
        ok += good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")
        if not good:
            print(f"          slot {slot}: got {f}")
            print(f"                  want {want}")
    print(f"\n  {ok}/{len(CASES)} passing")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
