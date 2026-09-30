"""needs_location — the hint that lets the client stop sending live coordinates.

Controls first: an ordinary booking must produce NO hint, or the client ends up
asking for GPS on rides that never needed it.
"""
import json, sys, time
import requests

import os
BASE = os.environ.get("PAYCHAT_URL", "http://127.0.0.1:8000")
R = int(time.time()) % 100000


def send(room, sender, text, mid=None):
    return requests.post(f"{BASE}/classify", json={
        "text": text, "room_id": room, "sender": str(sender),
        "message_id": mid or f"m{time.time_ns()}"}, timeout=60).json()


CASES = []


def case(name, turns, expect):
    CASES.append((name, turns, expect))


GPS  = lambda k: {"slot": k, "resolve": "gps", "place": None}
SAVE = lambda k, pl: {"slot": k, "resolve": "saved_place", "place": pl}
PART = lambda k, name, pl=None: {"slot": k, "resolve": "participant",
                                 "name": name, "place": pl}
# "your ..." - the other party, named by the pronoun rather than by name.
OTHER = lambda k, pl=None: {"slot": k, "resolve": "participant", "name": None,
                            "who": "other_party", "place": pl}
ASK = lambda k: {"slot": k, "resolve": "ask", "place": None}


# --- controls: nothing self-referential, so no hint ------------------------
# The client searches every place string with a local bias anyway; a hint here would
# carry no information it does not already have.
case("named pickup and destination", [(30, "book a cab from koramangala to marathahalli"),
                                      (31, "sure")], None)
case("destination only", [(30, "book me a cab to the airport"), (31, "ok booking")], None)
case("a business as the destination", [(30, "book a cab from koramangala to the adidas store"),
                                       (31, "sure")], None)
case("money, not ride", [(30, "send me 500"), (31, "sending now")], None)
case("ride discussed, never fires", [(30, "ola from my location is so expensive these days")], None)

# --- gps ---------------------------------------------------------------------
case("brahma: my location -> marathalli", [(30, "Book a cab from my location to marathalli"),
                                           (31, "Sure")],
     {"user_id": "30", "fields": [GPS("pickup")]})
case("my location -> a business", [(30, "book a cab from my location to the adidas store"),
                                   (31, "sure")],
     {"user_id": "30", "fields": [GPS("pickup")]})
case("a named place is never reported",
     [(30, "book a cab from koramangala to the adidas store"), (31, "sure")], None)
case("current location", [(30, "cab from my current location to hsr"), (31, "yeah booking")],
     {"user_id": "30", "fields": [GPS("pickup")]})

# --- phrasing variants of the same idea ---------------------------------------
case("my current loc", [(30, "book a cab from my current loc to hsr")],
     {"user_id": "30", "fields": [GPS("pickup")]})
case("my current spot", [(30, "book me a cab from my current spot to hsr")],
     {"user_id": "30", "fields": [GPS("pickup")]})
case("my current position", [(30, "cab from my current position to hsr")],
     {"user_id": "30", "fields": [GPS("pickup")]})
# Accommodation is deliberately NOT gps — the rider may not be there.
# Ruled 2026-09-02: home and office stay the ONLY saved place types. But silence was
# the wrong answer for the others - with no hint the client renders "My Pg" as though it
# were an address, which is the same bug Andril reported for "My Location". So these now
# report resolve="ask": unresolvable, the user has to pick an address.
case("my pg asks the user", [(30, "book a cab from my pg to hsr")],
     {"user_id": "30", "fields": [
         {"slot": "pickup", "phrase": "My Pg", "resolve": "ask", "place": None}]})
case("my hostel asks the user", [(30, "book a cab from my hostel to hsr")],
     {"user_id": "30", "fields": [
         {"slot": "pickup", "phrase": "My Hostel", "resolve": "ask", "place": None}]})

# --- saved places -------------------------------------------------------------
case("home pickup", [(30, "book a cab from my home to indiranagar"), (31, "on it")],
     {"user_id": "30", "fields": [SAVE("pickup", "home")]})
case("office pickup", [(30, "get me an uber from my office to whitefield"), (31, "sure")],
     {"user_id": "30", "fields": [SAVE("pickup", "office")]})
case("home as destination", [(30, "book a cab from btm to my home"), (31, "sure")],
     {"user_id": "30", "fields": [SAVE("destination", "home")]})

# --- the request itself, before anything fires --------------------------------
# The rider's app has to know at send time; a hint that only arrives on the fire is
# too late to attach anything to.
case("request names my location", [(30, "can you book me a cab from my location to tin factory")],
     {"user_id": "30", "fields": [GPS("pickup")]})
case("request names my home", [(30, "book me a cab from my home to koramangala")],
     {"user_id": "30", "fields": [SAVE("pickup", "home")]})
case("request with a named pickup asks nothing",
     [(30, "book me a cab from indiranagar to tin factory")], None)

# --- chatter that mentions a location but is not a booking --------------------
case("ride prices, not a booking", [(30, "ola from my location is so expensive these days")], None)
case("office chatter", [(30, "cabs from my office are always late")], None)
case("location sharing chatter", [(30, "my location sharing is broken again")], None)
case("stating where they are", [(30, "im at my home right now")], None)

# --- self-initiated: speaker is the actor -------------------------------------
case("self-initiated from my location", [(30, "booking an uber from my location to the airport")],
     {"user_id": "30", "fields": [GPS("pickup")]})

# --- phrases that were losing the slot entirely -------------------------------
# Both are in _GPS_PLACES, but spaCy tags them as adverbs so the parse produced no
# pobj, and "here" is in _VAGUE_PLACES on top of that: pickup came back None and the
# hint had nothing to attach to. Found by matrix-testing every location phrasing on
# 2026-09-30.
case("where i am", [(30, "can you book me a cab from where i am to the airport"), (31, "sure")],
     {"user_id": "30", "fields": [GPS("pickup")]})
case("from here", [(30, "can you book me a cab from here to the airport"), (31, "sure")],
     {"user_id": "30", "fields": [GPS("pickup")]})
case("where im at", [(30, "cab from where im at to hsr")],
     {"user_id": "30", "fields": [GPS("pickup")]})

# --- second person: the OTHER party's location --------------------------------
# Ruled by Akash 2026-09-30. The prompt opens on the booker, and the location wanted is
# the rider's - which the booker's own phone cannot read. Reported the same way a named
# participant is, with who="other_party" standing in for the name, because the resolver
# already knows how to ask a person for their position. user_id is still the SPEAKER of
# the phrase, so "the other party" means "whoever in this room is not user_id".
case("your current location is the other party",
     [(31, "i will book you a cab from your current location to the airport")],
     {"user_id": "31", "fields": [OTHER("pickup")]})
case("your location, shorter",
     [(31, "let me book you a cab from your location to the airport")],
     {"user_id": "31", "fields": [OTHER("pickup")]})
case("your place is the other party's home",
     [(31, "i will book you a cab from your place to the airport")],
     {"user_id": "31", "fields": [OTHER("pickup", "home")]})
case("your office is the other party's office",
     [(31, "ill book you a cab from your office to the airport")],
     {"user_id": "31", "fields": [OTHER("pickup", "office")]})
# The rider asking someone else to read THEIR location, rather than the booker offering.
case("rider asks from your location",
     [(30, "can you book a cab from your location to the airport"), (31, "sure")],
     {"user_id": "30", "fields": [OTHER("pickup")]})
# "send a cab TO your location" puts it in destination. The slot is arguable - the cab
# collects there, so it reads like a pickup - but the resolution is the same either way
# and mis-slotting it is a parse question, tracked separately from the hint.
case("a cab sent to your location", [(31, "let me send a cab to your current location")],
     {"user_id": "31", "fields": [OTHER("destination")]})

# --- third person, named: unchanged, but the phrase is now readable ------------
case("a participant's current location",
     [(30, "can someone book a cab from my current location to Brahma's current location")],
     {"user_id": "30", "fields": [GPS("pickup"), PART("destination", "Brahma")]})
# spaCy splits the clitic and the subtree join put the space back, so the field the user
# edits used to read "Andril 'S Place".
case("a participant's place, rendered readably",
     [(30, "can you book a cab from andril's place to the airport")],
     {"user_id": "30", "fields": [
         {"slot": "pickup", "phrase": "Andril's Place", "resolve": "participant",
          "name": "Andril", "place": "home"}]})

# --- third person, pronoun: nobody to ask --------------------------------------
# "his current location" used to come back as participant name "Hi": the regex accepts a
# bare trailing s as the possessive, so the pronoun was captured one letter short and the
# client went looking for a participant called Hi. There is nobody for the resolver to
# ask here, so `ask` - which at least stops the literal text being rendered as an address.
case("his location has nobody to ask",
     [(30, "can you book a cab from his current location to the airport")],
     {"user_id": "30", "fields": [ASK("pickup")]})
case("her location has nobody to ask",
     [(30, "can you book a cab from her place to the airport")],
     {"user_id": "30", "fields": [ASK("pickup")]})
case("their location has nobody to ask",
     [(30, "can you book a cab from their location to the airport")],
     {"user_id": "30", "fields": [ASK("pickup")]})
case("our location has nobody to ask",
     [(30, "can you book us a cab from our current location to the airport")],
     {"user_id": "30", "fields": [ASK("pickup")]})

# --- whose location, on an accepted request -----------------------------------
# The one that bites: the sheet opens for 31, but "my" was 30's. A client resolving
# "My Current Location" against its own GPS books the cab from the wrong person.
case("an accepted request keeps the asker's location",
     [(30, "book a cab from my current location to marathalli"), (31, "sure")],
     {"user_id": "30", "fields": [GPS("pickup")]})


fails = 0
for i, (name, turns, expect) in enumerate(CASES):
    room = f"dm_{R + i}_{R + i + 1}"
    det = {}
    for sender, text in turns:
        det = send(room, sender, text) or {}
    got = det.get("needs_location")
    ok = True
    if expect is None:
        ok = got is None
    else:
        ok = bool(got) and got.get("user_id") == expect["user_id"] and \
             len(got.get("fields", [])) == len(expect["fields"]) and \
             all(all(f.get(k) == v for k, v in e.items())
                 for f, e in zip(got["fields"], expect["fields"]))
    if not ok:
        fails += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"          fired    : {det.get('intents')}")
        print(f"          slots    : {det.get('slots')}")
        print(f"          expected : {expect}")
        print(f"          got      : {got}")

print(f"\n  {len(CASES) - fails}/{len(CASES)} passed")
sys.exit(1 if fails else 0)
