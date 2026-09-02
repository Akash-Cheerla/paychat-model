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
