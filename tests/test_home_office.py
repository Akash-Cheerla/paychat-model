"""home and office, as deep as it goes - the two place types a profile holds today.

Written 2026-10-01 after Akash asked for every ride-location case to be tested before
pushing: "just focus on home and office words for now, test as much as possible". It found
three real defects, all of which this file now guards:

  * "Rupesh New Flat" resolved to his HOME. The bare "<Name> <label>" form was split
    greedily as name "Rupesh New" + label "Flat", and "flat" reaches home through the
    alias table - a wrong saved address on a booking, which is the one failure the design
    exists to prevent. Splits are now tried longest-label-first.
  * "GT Home" produced nothing: the bare form required a name of three characters, which
    blocks a two-letter nickname. A name that short can only match the roster exactly.
  * "gowthams wrk" fell to the picker. "wrk" is one edit from "work", which means office,
    but the person stores "office" - so the fuzzy pass now also matches the ALIAS words.

Section 5 and 6 matter as much as the rest: "home depot", "office park", "i work from home
today" and "brahmas office has the best coffee" must produce no field at all. Opening the
place vocabulary is only safe while those stay quiet.

    python tests/test_home_office.py --url http://127.0.0.1:8971/detect
"""
import argparse
import io
import json
import sys
import time

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Gowtham has BOTH. Brahma has only office. Rupesh has only home. Andril has neither.
# That spread is what makes "not saved" testable in every direction.
R = [
    {"id": "10", "name": "Gowtham", "nickname": "GT", "places": ["home", "office"]},
    {"id": "20", "name": "Brahma", "nickname": "BR", "places": ["office"]},
    {"id": "53", "name": "Rupesh", "nickname": None, "places": ["home"]},
    {"id": "4", "name": "Andril", "nickname": None, "places": []},
]
DM = [
    {"id": "30", "name": "Rupesh", "nickname": None, "places": ["home"]},
    {"id": "31", "name": "Yash", "nickname": None, "places": ["home", "office"]},
]

HOME = lambda u: {"resolve": "saved_place", "place": "home", "user_id": u}
OFFICE = lambda u: {"resolve": "saved_place", "place": "office", "user_id": u}
ASK = lambda **k: dict({"resolve": "ask"}, **k)

C = []
def case(label, roster, turns, slot, expect, kind="group"):
    C.append((label, kind, roster, turns, slot, expect))


# ---- 1. the speaker's own (sender 10 = Gowtham, has both) -------------------------
for w in ["home", "my home", "my house", "my place", "my flat", "my apartment",
          "my residence", "my address", "my addr"]:
    case(f"1: own home via '{w}'", R, [("10", f"book a cab from {w} to rv road")],
         "pickup", HOME("10"))
for w in ["office", "my office", "work", "my work", "my workplace"]:
    case(f"1: own office via '{w}'", R, [("10", f"book a cab from {w} to rv road")],
         "pickup", OFFICE("10"))

# ---- 2. a named member, every possessive form -------------------------------------
for w in ["gowtham's home", "gowthams home", "Gowtham Home", "gowtham’s home",
          "GOWTHAM'S HOME", "gowtham's house", "gowthams house", "Gowtham House",
          "gowthams flat", "gowthams apartment", "gowthams residence",
          "gowthams address", "gowthams addr", "Gowtham Address",
          "gowtham reddys home", "Gowtham Reddy Home", "GTs home", "GT's home",
          "GT Home"]:
    case(f"2: Gowtham home via '{w}'", R,
         [("53", f"can someone book a cab from {w} to rv road")], "pickup", HOME("10"))
for w in ["gowtham's office", "gowthams office", "Gowtham Office", "gowthams work",
          "gowthams workplace", "Gowtham Workplace", "GTs office", "GT Office"]:
    case(f"2: Gowtham office via '{w}'", R,
         [("53", f"can someone book a cab from {w} to rv road")], "pickup", OFFICE("10"))
for w in ["brahmas office", "Brahma Office", "brahmas work", "BRs office", "BR Office"]:
    case(f"2: Brahma office via '{w}'", R,
         [("53", f"can someone book a cab from {w} to rv road")], "pickup", OFFICE("20"))
for w in ["rupeshs home", "Rupesh Home", "rupeshs house", "rupeshs address"]:
    case(f"2: Rupesh home via '{w}'", R,
         [("10", f"can someone book a cab from {w} to rv road")], "pickup", HOME("53"))

# ---- 3. whose it is --------------------------------------------------------------
case("3: your home in a DM is the other party", DM,
     [("30", "i will book you a cab from your home to the airport")], "pickup",
     HOME("31"), kind="dm")
case("3: your office in a DM", DM,
     [("30", "i will book you a cab from your office to the airport")], "pickup",
     OFFICE("31"), kind="dm")
case("3: your home in a group is the picker", R,
     [("10", "i will book you a cab from your home to the airport")], "pickup",
     ASK(who="other_party"))
case("3: his home -> picker, nobody named", R,
     [("53", "book a cab from his home to rv road")], "pickup", ASK())
case("3: their office -> picker", R,
     [("53", "book a cab from their office to rv road")], "pickup", ASK())
case("3: destination home of a member", R,
     [("53", "book a cab from rv road to gowthams home")], "destination", HOME("10"))
case("3: destination office of a member", R,
     [("53", "book a cab from rv road to Brahma Office")], "destination", OFFICE("20"))
case("3: destination own home", R,
     [("10", "book a cab from rv road to my home")], "destination", HOME("10"))
case("3: accepted request keeps the ASKER's home", R,
     [("53", "can someone book a cab from my home to rv road"), ("4", "i can book it")],
     "pickup", HOME("53"))
case("3: accepted request keeps the NAMED member", R,
     [("53", "can someone book a cab from gowthams office to rv road"),
      ("4", "i can book it")], "pickup", OFFICE("10"))

# ---- 4. not saved -> the picker, and never the other saved place ------------------
case("4: Brahma has no home -> ask, NOT his office", R,
     [("53", "book a cab from brahmas home to rv road")], "pickup",
     ASK(place="home", user_id="20"))
case("4: Brahma bare form, no home -> ask", R,
     [("53", "book a cab from Brahma Home to rv road")], "pickup", None)
case("4: Rupesh has no office -> ask, NOT his home", R,
     [("10", "book a cab from rupeshs office to rv road")], "pickup",
     ASK(place="office", user_id="53"))
case("4: Andril has neither (home)", R,
     [("53", "book a cab from andrils home to rv road")], "pickup",
     ASK(place="home", user_id="4"))
case("4: Andril has neither (office)", R,
     [("53", "book a cab from andrils office to rv road")], "pickup",
     ASK(place="office", user_id="4"))
case("4: speaker 20 has no home of his own", R,
     [("20", "book a cab from my home to rv road")], "pickup",
     ASK(place="home", user_id="20"))
case("4: speaker 53 has no office of his own", R,
     [("53", "book a cab from my office to rv road")], "pickup",
     ASK(place="office", user_id="53"))

# ---- 5. real places that CONTAIN the words -------------------------------------
for w in ["home depot", "the home depot", "office depot", "homebase",
          "home town hotel", "office park", "the office canteen",
          "ikea home centre", "home saluja"]:
    case(f"5: '{w}' is a place, not a person", R,
         [("53", f"book a cab from {w} to rv road")], "pickup", None)
case("5: a member named nothing like it", R,
     [("53", "book a cab from koramangala office park to rv road")], "pickup", None)

# ---- 6. the words used non-locationally ------------------------------------------
for text in ["i work from home today so no cab needed",
             "the office is closed tomorrow",
             "im at home right now",
             "take me home" ,
             "gowthams home is near the metro",
             "brahmas office has the best coffee",
             "my home internet is down",
             "office politics again"]:
    case(f"6: '{text[:36]}'", R, [("53", text)], "pickup", None)

# ---- 7. spelling, on the name and on the word -----------------------------------
for w in ["gouthams home", "gowtams home", "gauthams home", "Goutham Home",
          "gowthams hoem", "gowthams hme", "gowthams ofice", "gowthams offce",
          "gowthams wrk", "bramhas office", "Bramha Office"]:
    case(f"7: '{w}'", R, [("53", f"book a cab from {w} to rv road")], "pickup",
         HOME("10") if ("ho" in w.lower() or "hme" in w.lower()) else
         (OFFICE("20") if "ramha" in w.lower() or "bramha" in w.lower() else OFFICE("10")))

# ---- 8. sentence shapes ----------------------------------------------------------
case("8: both ends are members", R,
     [("53", "book a cab from gowthams home to brahmas office")], "pickup", HOME("10"))
case("8: both ends are members (dest)", R,
     [("53", "book a cab from gowthams home to brahmas office")], "destination",
     OFFICE("20"))
case("8: own home to member office", R,
     [("10", "book a cab from my home to brahmas office")], "pickup", HOME("10"))
case("8: no destination", R, [("53", "book a cab from gowthams home")], "pickup",
     HOME("10"))
case("8: trailing question mark", R,
     [("53", "can someone book me a cab from Gowtham Home to JP nagar ?")], "pickup",
     HOME("10"))
case("8: extra words around it", R,
     [("53", "hey guys pls book me a cab from gowthams home to jp nagar asap")],
     "pickup", HOME("10"))
case("8: from home to office, same speaker", R,
     [("10", "book a cab from my home to my office")], "pickup", HOME("10"))
case("8: from home to office, destination", R,
     [("10", "book a cab from my home to my office")], "destination", OFFICE("10"))
case("8: no roster -> participant, unchanged", None,
     [("53", "book a cab from gowthams home to rv road")], "pickup",
     {"resolve": "participant", "name": "Gowtham", "place": "home"})


def field(hint, slot):
    for f in (hint or {}).get("fields") or []:
        if f.get("slot") == slot:
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8971/detect")
    a = ap.parse_args()
    tag = int(time.time())
    s = requests.Session()
    failed, sect = [], None
    for ci, (label, kind, roster, turns, slot, want) in enumerate(C):
        head = label.split(":")[0]
        if head != sect:
            sect = head
            print(f"\n--- section {sect}")
        room = (f"dm_{tag}{ci}_{tag}{ci+1}" if kind == "dm" else f"grp_{tag}_{ci}")
        d = {}
        for j, (sender, text) in enumerate(turns):
            body = {"text": text, "room_id": room, "sender": sender,
                    "message_id": f"{tag}_{ci}_{j}",
                    "participants": len(roster) if roster else 4}
            if roster is not None:
                body["roster"] = roster
            d = s.post(a.url, json=body, timeout=90).json()
        f = field(d.get("needs_location"), slot)
        ok = (f is None) if want is None else (bool(f) and all(f.get(k) == v for k, v in want.items()))
        if not ok:
            failed.append(label)
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            sl = {k: v for k, v in (d.get("slots") or {}).items() if v}
            print(f"          want {want}")
            print(f"          got  {f}")
            print(f"          slots {json.dumps(sl, ensure_ascii=False)}")
    print(f"\n  {len(C) - len(failed)}/{len(C)} passing")
    if failed:
        print("\n  failures:")
        for x in failed:
            print("   -", x)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
