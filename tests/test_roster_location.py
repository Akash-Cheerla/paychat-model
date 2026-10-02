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


# Labels beyond home/office. Akash, 2026-10-01: "rn we have house and office for each user
# but in future they can keep whatever ... all we have to do is match them". So the place
# word is no longer a fixed list - the member's own `places` decides, which also means an
# unsaved label is the picker rather than a guess.
KEEPERS = [
    {"id": "10", "name": "Gowtham", "nickname": "GT", "places": ["home", "gym"]},
    {"id": "20", "name": "Brahma", "nickname": None, "places": ["office", "pg"]},
    {"id": "53", "name": "Rupesh", "nickname": None, "places": ["home", "new flat"]},
    {"id": "4", "name": "Andril", "nickname": None, "places": []},
]

CASES += [
    ("a saved gym resolves to the gym",
     "group", KEEPERS, [("53", "can someone book me a cab from gowthams gym to rv road")],
     "pickup", {"resolve": "saved_place", "place": "gym", "user_id": "10"}),

    ("a saved pg resolves for a different member",
     "group", KEEPERS, [("53", "can someone book me a cab from brahmas pg to indiranagar")],
     "pickup", {"resolve": "saved_place", "place": "pg", "user_id": "20"}),

    ("a two-word label resolves",
     "group", KEEPERS, [("10", "can someone book me a cab from rupeshs new flat to the airport")],
     "pickup", {"resolve": "saved_place", "place": "new flat", "user_id": "53"}),

    ("a label that member has NOT saved is the picker, owner known",
     "group", KEEPERS, [("53", "can someone book me a cab from gowthams pg to rv road")],
     "pickup", {"resolve": "ask", "place": "pg", "user_id": "10"}),

    # Aliases still work in both directions, so the sentence need not use the word the
    # profile stored. The reported place is the STORED spelling - that is what the caller
    # looks up.
    ("house in the sentence finds a stored home",
     "group", KEEPERS, [("53", "can someone book me a cab from gowthams house to rv road")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "10"}),
    ("work in the sentence finds a stored office",
     "group", KEEPERS, [("53", "can someone book me a cab from brahmas work to rv road")],
     "pickup", {"resolve": "saved_place", "place": "office", "user_id": "20"}),
    ("address in the sentence finds a stored home",
     "group", KEEPERS, [("10", "can someone book me a cab from rupeshs address to rv road")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "53"}),

    ("a nickname resolves an open label",
     "group", KEEPERS, [("53", "can someone book me a cab from GTs gym to rv road")],
     "pickup", {"resolve": "saved_place", "place": "gym", "user_id": "10"}),

    # The old single-token capture read this as a member called "reddy" and matched nobody.
    ("a first and last name still matches on the first name",
     "group", KEEPERS, [("53", "can someone book me a cab from gowtham reddys home to rv road")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "10"}),

    ("a member's current location is still their device",
     "group", KEEPERS, [("53", "can someone book me a cab from gowthams current location to rv road")],
     "pickup", {"resolve": "gps", "place": None, "user_id": "10"}),

    # The speaker's own open label, same rule.
    ("my gym resolves when the speaker saved one",
     "group", KEEPERS, [("10", "can someone book me a cab from my gym to rv road")],
     "pickup", {"resolve": "saved_place", "place": "gym", "user_id": "10"}),
    ("my pg is the picker when the speaker saved none",
     "group", KEEPERS, [("10", "can someone book me a cab from my pg to rv road")],
     "pickup", {"resolve": "ask", "user_id": "10"}),
]


# Spelling. Ruled by Akash 2026-10-01: resolve a near-miss when exactly one member is close
# and the next closest is clearly further. The common case is not a typing error - an Indian
# name has several romanisations and whoever types picks theirs, not the one in the other
# person's profile.
CASES += [
    ("a romanisation variant of the name resolves",
     "group", KEEPERS, [("53", "book a cab from gouthams home to rv road")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "10"}),
    ("a dropped letter in the name resolves",
     "group", KEEPERS, [("53", "book a cab from gowtams home to rv road")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "10"}),
    ("two substitutions still resolve",
     "group", KEEPERS, [("53", "book a cab from gauthams home to rv road")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "10"}),
    ("a transposition resolves (brahma / bramha)",
     "group", KEEPERS, [("53", "book a cab from bramhas office to rv road")],
     "pickup", {"resolve": "saved_place", "place": "office", "user_id": "20"}),
    # The possessive regex takes the bare s as the marker, so "rupes home" hands over the
    # name "rupe" - the matcher has to try it with the s put back.
    ("a name the possessive ate a letter from resolves",
     "group", KEEPERS, [("10", "book a cab from rupes home to rv road")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "53"}),

    # Labels, against that person's own short list.
    ("a misspelt label finds the saved one",
     "group", KEEPERS, [("53", "book a cab from gowthams hoem to rv road")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "10"}),
    ("a label missing a letter finds the saved one",
     "group", KEEPERS, [("53", "book a cab from brahmas ofice to rv road")],
     "pickup", {"resolve": "saved_place", "place": "office", "user_id": "20"}),
    ("a two-letter label still finds the gym",
     "group", KEEPERS, [("53", "book a cab from gowthams gm to rv road")],
     "pickup", {"resolve": "saved_place", "place": "gym", "user_id": "10"}),
    ("a misspelt two-word label resolves",
     "group", KEEPERS, [("10", "book a cab from rupeshs new flt to rv road")],
     "pickup", {"resolve": "saved_place", "place": "new flat", "user_id": "53"}),
    # "hom" is two edits from the position word "loc", and one from their saved "home".
    # The saved place has to win, or a misspelling turns into a device lookup.
    ("a saved place outranks a misspelt position word",
     "group", KEEPERS, [("53", "book a cab from gowthams hom to rv road")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "10"}),
]

# The margin rule: two members with similar names, so NOBODY is chosen. A wrong person's
# home in a cab booking is worse than a picker, and this is the case that guarantees it.
TWINS = [
    {"id": "10", "name": "Rupesh", "nickname": None, "places": ["home"]},
    {"id": "11", "name": "Rupesha", "nickname": None, "places": ["home"]},
    {"id": "53", "name": "Gowtham", "nickname": None, "places": ["home"]},
]
CASES += [
    ("a near-miss between two similar names resolves to nobody",
     "group", TWINS, [("53", "book a cab from rupes home to rv road")],
     "pickup", {"resolve": "ask", "user_id": None}),
    ("an exact name still wins when a similar one exists",
     "group", TWINS, [("53", "book a cab from rupeshs home to rv road")],
     "pickup", {"resolve": "saved_place", "place": "home", "user_id": "10"}),
]

# A landmark that happens to be near a member's name must NOT become that member's place:
# the name AND the label both have to land for an open label.
NEARBY_NAME = [
    {"id": "10", "name": "Dominic", "nickname": None, "places": ["home", "office"]},
    {"id": "53", "name": "Gowtham", "nickname": None, "places": ["home"]},
]

# Things that fit "<word>'s <word>" but are NOT a person's saved place. Each must produce
# NO field: a landmark belongs in Places, and a picker would be worse than a search.
NO_FIELD = [
    ("a hospital is a place, not a member",
     "can someone book me a cab from st johns hospital to rv road"),
    ("a shop is a place, not a member",
     "can someone book me a cab from koramangala to the adidas store"),
    ("kinship is never a room member",
     "can someone book me a cab from moms house to rv road"),
    ("a possessive that is not a place at all",
     "can someone book a cab to rv road, gowthams idea was to leave at 9"),
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
    total = len(CASES)
    for name, text in NO_FIELD + [
        # "domino" is two edits from the member "Dominic", and he has no "pizza" saved, so
        # the open label does not land and the shop stays a Places search.
        ("a shop whose name is close to a member's", "book a cab from dominos pizza to rv road"),
    ]:
        total += 1
        room = f"grp_{tag}_nf{total}"
        roster = NEARBY_NAME if "domino" in text else KEEPERS
        d = requests.post(a.url, timeout=90, json={
            "text": text, "room_id": room, "sender": "53", "message_id": f"{tag}_nf{total}",
            "participants": len(roster), "roster": roster}).json()
        got = [f for f in (d.get("needs_location") or {}).get("fields") or []]
        good = not got
        ok += good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")
        if not good:
            print(f"          expected no field, got {got}")
            print(f"          slots {d.get('slots')}")

    print(f"\n  {ok}/{total} passing")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
