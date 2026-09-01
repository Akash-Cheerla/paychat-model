"""Every bug reported from real use, replayed as a conversation.

One case per thing that was actually wrong in a real chat or a screenshot — not
invented edge cases. If a case here goes red, a bug we told someone was fixed has come
back. Slot controls live in test_ride_slots.py; this file is the behaviour side.

Each case is (name, room_kind, conversation, check, expectation) where room_kind is
"dm" or "group" — a DM room id must look like dm_<lo>_<hi> or ambient matching never
runs, which silently turns most of these green for the wrong reason.

Run:  python test_reported_bugs.py [--url http://127.0.0.1:8005/detect]
"""
import argparse, itertools, sys, time

import requests

# Clock-seeded, never a constant: room state lives on the server for 4 hours, so a
# fixed start makes every re-run replay into rooms that still hold the previous run's
# fires. That silently turned two green controls red once already.
_ids = itertools.count(int(time.time() * 1000) % 1_000_000_000)


def play(url, kind, convo):
    """Replay a conversation. Returns one record per turn."""
    n = next(_ids)
    room = f"dm_{n}_{n + 7}" if kind == "dm" else f"grp_{n}"
    out = []
    for i, (who, text) in enumerate(convo):
        r = requests.post(url, json={"text": text, "room_id": room, "sender": who,
                                     "message_id": f"m{i}"}, timeout=180).json()
        cs = r.get("conversation_state") or {}
        tb = cs.get("triggered_by") or {}
        out.append({
            "sender": who,
            "text": text,
            "fired": [x for x in (r.get("intents") or []) if x in ("money", "ride")],
            "slots": tb.get("slots") or {},
            "score": round((cs.get("scores") or {}).get("ride", 0), 3),
        })
    return out


def fires(rec):
    return [t for t in rec if t["fired"]]


def slot(t, k):
    return (t["slots"] or {}).get(k)


def has(v, *words):
    s = (v or "").lower()
    return all(w in s for w in words)


CASES = [
    # ── Brahma's booking chat, 2026-08-10 screenshots ──
    ("brahma: address on its own lines keeps the street", "dm",
     [("G", "Can you book a cab for me\nVakil garden city, Kanakapura road Bangalore\n"
            "To Bengaluru international airport terminal 2"), ("B", "Sure")],
     lambda r: len(fires(r)) == 1 and has(slot(fires(r)[0], "pickup"), "vakil")
               and has(slot(fires(r)[0], "destination"), "airport"),
     "one prompt, pickup keeps 'vakil', destination keeps 'airport'"),

    ("brahma: 'my location' is kept for the user to edit", "dm",
     [("G", "Pls book a cab from my location to terminal 2"), ("B", "Sure")],
     lambda r: len(fires(r)) == 1 and has(slot(fires(r)[0], "pickup"), "location"),
     "pickup 'My Location', not blank and not guessed"),

    ("brahma: address stated in a later message fills the pickup", "dm",
     [("G", "Pls book a cab to the airport"), ("B", "Sure"),
      ("G", "My address is vakil garden city, Kanakapura road, talaghattapura")],
     lambda r: fires(r) and any(has(slot(t, "pickup"), "vakil") or
                                has(slot(t, "destination"), "airport") for t in fires(r)),
     "the stated address reaches the booking"),

    # ── the stale-trip bug: a fresh request showed the PREVIOUS trip ──
    ("stale: a new unparseable request does not show the old trip", "dm",
     [("A", "book a cab from jp nagar to Banashankari"), ("B", "Sure"),
      ("A", "book smth for me"), ("B", "Sure")],
     lambda r: len(fires(r)) < 2 or not slot(fires(r)[1], "destination"),
     "second prompt blank, NOT jp nagar -> banashankari"),

    ("stale: a new request with its own places wins", "dm",
     [("A", "book a cab from jp nagar to banashankari"), ("B", "Sure"),
      ("A", "book a cab from indiranagar to whitefield"), ("B", "Sure")],
     lambda r: len(fires(r)) == 2 and has(slot(fires(r)[1], "destination"), "whitefield"),
     "second prompt indiranagar -> whitefield"),

    # ── London -> Windsor: the booker proposed a different destination and the
    #    rider never agreed, but the prompt followed the booker ──
    ("only the rider changes their own destination", "dm",
     [("A", "book me a cab from london to windsor"), ("B", "isnt slough closer"),
      ("B", "booking it now")],
     lambda r: fires(r) and has(slot(fires(r)[-1], "destination"), "windsor"),
     "destination stays windsor — the rider never agreed to slough"),

    ("one trip produces one prompt, not two", "dm",
     [("A", "book me a cab to koramangala"), ("B", "sure"), ("B", "booking it now")],
     lambda r: len(fires(r)) == 1,
     "a restatement of the same booking is not a second prompt"),

    # ── payer targeting: the prompt used to go to whoever spoke last ──
    ("an accepted offer charges the offerer, not the accepter", "dm",
     [("A", "shall i send you 500 for the tickets"), ("B", "sure")],
     lambda r: len(fires(r)) == 1 and str(slot(fires(r)[0], "amount") or "").find("500") >= 0,
     "one money prompt carrying 500"),

    ("a counter-offer prompts for the agreed amount", "dm",
     [("A", "can u lend me 2000"), ("B", "i can only do 1000"), ("A", "cool send it")],
     lambda r: fires(r) and "1000" in str(slot(fires(r)[-1], "amount") or ""),
     "prompt shows 1000, not the 2000 first asked for"),

    # ── rides are ride-hailing only ──
    ("a friend offering their own car is not a ride", "dm",
     [("A", "can you drop me at the airport tomorrow morning?"),
      ("B", "sure i'll pick you up at 6")],
     lambda r: not fires(r),
     "no prompt — a lift is not a booking"),

    ("a rejection cancels the request", "dm",
     [("A", "book me a cab to koramangala"), ("B", "cant man im in a meeting"),
      ("A", "no worries")],
     lambda r: not fires(r),
     "no prompt while the request stands refused"),

    ("a refusal reversed later still books", "dm",
     [("A", "book me a cab to koramangala"), ("B", "cant man im in a meeting"),
      ("A", "no worries"), ("B", "actually meeting ended, booking it now")],
     lambda r: len(fires(r)) == 1 and has(slot(fires(r)[0], "destination"), "koramangala"),
     "one prompt on the revival, destination recovered"),

    # ── groups ──
    ("group: two bookers both get the destination", "group",
     [("A", "we're 7 people, need 2 cabs to the airport"),
      ("B", "i'll book one"), ("C", "ok i'll book the other one")],
     lambda r: len(fires(r)) == 2 and all(has(slot(t, "destination"), "airport")
                                          for t in fires(r)),
     "both prompts carry airport"),

    ("group: a split names a per-person share", "group",
     [("A", "dinner was 3000, split 3 ways"), ("B", "sending my part now")],
     lambda r: fires(r) and "1000" in str(slot(fires(r)[-1], "amount") or ""),
     "prompt shows 1000, not the 3000 total"),

    ("group: an unknown headcount blanks rather than guesses", "group",
     [("A", "dinner was 3000, lets split it"), ("B", "sending my part now")],
     lambda r: fires(r) and "3000" not in str(slot(fires(r)[-1], "amount") or ""),
     "prompt does NOT pre-fill the full 3000"),

    # ── completed actions never re-prompt (FIRING_RULE 3a) ──
    ("an already-completed payment does not prompt", "dm",
     [("A", "did you send the 500?"), ("B", "yeah sent it yesterday")],
     lambda r: not fires(r),
     "no prompt — the money already moved"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8005/detect")
    a = ap.parse_args()
    bad = []
    for name, kind, convo, check, expect in CASES:
        rec = play(a.url, kind, convo)
        try:
            ok = check(rec)
        except Exception as e:                       # a check that throws is a failure
            ok, expect = False, f"{expect}  [check raised {e!r}]"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            bad.append(name)
            for t in rec:
                mark = "FIRE" if t["fired"] else "  . "
                print(f"          {mark} {t['sender']:3} {t['text'][:52]:54} {t['slots']}")
            print(f"          expect {expect}")
    print(f"\n  {len(CASES) - len(bad)}/{len(CASES)} passing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
