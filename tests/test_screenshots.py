"""The chats from the dogfood screenshots, replayed turn for turn.

Reconstructed from the screenshots shared 2026-08-07 to 2026-08-11. Typos, repeated
messages and the testers' own commentary ("Got intent for sending asap", "Not getting
an intent at all") are all kept verbatim — those lines were real messages in the room,
the model scored them, and in one case a commentary line was itself what produced a
duplicate prompt.

Each conversation is replayed whole. Every one of these bugs was about what the app
carried over from EARLIER messages, so a per-turn test would miss all of them.

Only money and ride are surfaced, so the calendar icon on "Post Fri deployment next
week" and the flight icon on "the Toronto airport" cannot recur — those intents are
switched off. They are noted where they appeared but not asserted on.

Run:  python test_screenshots.py [--url http://127.0.0.1:8002/detect]
"""
import argparse, itertools, sys, time

import requests

_ids = itertools.count(int(time.time() * 1000) % 1_000_000_000)


def play(url, kind, convo):
    n = next(_ids)
    room = f"dm_{n}_{n + 11}" if kind == "dm" else f"grp_{n}"
    out = []
    for i, (who, text) in enumerate(convo):
        r = requests.post(url, json={"text": text, "room_id": room, "sender": who,
                                     "message_id": f"m{i}"}, timeout=180).json()
        cs = r.get("conversation_state") or {}
        out.append({"who": who, "text": text,
                    "fired": [x for x in (r.get("intents") or []) if x in ("money", "ride")],
                    "slots": (cs.get("triggered_by") or {}).get("slots") or {}})
    return out


def money(p):
    return [t for t in p if "money" in t["fired"]]


def rides(p):
    return [t for t in p if "ride" in t["fired"]]


def amt(t):
    return str(t["slots"].get("amount") or "")


def dest(t):
    return str(t["slots"].get("destination") or "").lower()


def pick(t):
    return str(t["slots"].get("pickup") or "").lower()


SHOTS = [
    # ── Gowtham, 12:01-12:08. Reported: "Did not get any intent", repeatedly, for a
    #    $20 debt that was stated in words ("twenty bucks") then in digits.
    ("Gowtham — $20 owed, then the cab (12:01-12:08)", "dm",
     [("G", "Hi Akash"),
      ("A", "Can you book me a cab from Pearson airport, Mississauga to 47 shepton way , Scarborough, Ontario ?"),
      ("G", "Will come to that. First of all, let's discuss the payment that I owe you"),
      ("A", "How much do you owe me?"),
      ("G", "I remember it as twenty bucks"),
      ("A", "Can you send that now please?"),
      ("G", "Post Fri deployment next week, let's discuss future features for See"),
      ("G", "Sure"),
      ("G", "Will send the cash now"),
      ("G", "Did not get any intent Akash"),
      ("A", "Please send me that 20$ you owe me"),
      ("G", "Only the 'this message has a compliance issue' notification"),
      ("G", "Okay. Will send"),
      ("G", "Not getting an intent at all"),
      ("G", "Shall I send 20 dollars now"),
      ("A", "Yes please send"),
      ("G", "Okay"),
      ("G", "Still no intent"),
      ("G", "Will book that cab for you now"),
      ("A", "Yes please")],
     [("a money prompt appears for the 20 owed", lambda p: bool(money(p))),
      ("it is 20, not some other figure",
       lambda p: not money(p) or any("20" in amt(t) for t in money(p))),
      # KNOWN RED, 2026-08-11 — accepted for now, tracked not hidden.
      #
      # Was four money sheets, now three. Two of the three are legitimate: A asked for
      # the 20 three separate times ("Can you send that now please?", "Please send me
      # that 20$ you owe me") and G offered once ("Shall I send 20 dollars now"), and
      # a fresh request reopens the intent by design. The one that is still wrong is
      # the last, which answers "let's discuss the payment that I owe you" (raw 0.73)
      # — meta-discussion recorded as a request. _AMOUNT_INQUIRY removed the "how much
      # do you owe me?" record; discussion-framing would need its own guard.
      #
      # Kept red so the count is visible if it ever climbs again.
      # Cause is in the request-recording branch, not the dedup: any message whose
      # RAW money score crosses 0.606 is stored as an open request, and questions
      # about the debt score high — "How much do you owe me?" is 0.787, "let's
      # discuss the payment that I owe you" is 0.734. Each spurious request gives a
      # later commitment something extra to answer, so "Sure" / "Will send the cash
      # now" / "Okay. Will send" each consume a different one and each opens a sheet.
      # has_open cannot filter it: it correctly reports an unanswered request that
      # should never have been recorded. Fix belongs where requests are recorded —
      # a question about an amount is not a request for it.
      ("one payment, one prompt", lambda p: len(money(p)) == 1),
      ("a ride prompt appears for the cab", lambda p: bool(rides(p))),
      ("the cab is Pearson -> 47 shepton way",
       lambda p: not rides(p) or any("shepton" in dest(t) or "scarborough" in dest(t)
                                     for t in rides(p)))]),

    # ── Yash, 12:24-12:43. THE reported bug: after a 100 -> 60 negotiation settled,
    #    a NEW request for 40 (lunch) prompted for 60 — the old figure bled across.
    ("Yash — lunch 40 must not become 60 (12:24-12:43)", "dm",
     [("Y", "Hey Akash"),
      ("Y", "Can you see my message ?"),
      ("A", "Hey Yash!"),
      ("A", "Can you book me a cab from Pearson airport, Mississauga to 47 shepton way , Scarborough, Ontario ?"),
      ("Y", "Can you book me a cab from Pearson airport, Mississauga to 47 shepton way , Scarborough, Ontario ?"),
      ("Y", "Sure I can"),
      ("Y", "Got the intent and it worked fine"),
      ("Y", "Can you send me hundred bucks on paypal ?"),
      ("A", "I can only do 60$ man, is that fine?"),
      ("Y", "Sure but send it now as i need it urgently. Remaining $40 I'll arrange"),
      ("A", "Sure sending now"),
      ("A", "Can you send 40$ for my lunch please"),
      ("Y", "Have you watched the latest spiderman movie ?"),
      ("A", "Yes , it was brilliant . Saw it in imax , did you?"),
      ("A", "What about odyssey?"),
      ("Y", "I'm planning to..this weekend"),
      ("Y", "Odyssey was superb. Have you watched it ?"),
      ("A", "Yes yes had to read some stuff prior to watch it"),
      ("Y", "Ohh yeah about the lunch money let send you now"),
      ("A", "Yes please"),
      ("Y", "I completely forgot about the lunch money you needed..let me send you now"),
      ("A", "Yes please")],
     [("the lunch payment prompts for 40, NOT the 60 from the earlier deal",
       lambda p: bool(money(p)) and "40" in amt(money(p)[-1])),
      ("60 does not appear on the lunch prompt",
       lambda p: not money(p) or "60" not in amt(money(p)[-1]))]),

    # ── Gowtham, 2:16. "Sending asap" fired, and then the tester's own commentary
    #    "Got intent for sending asap" fired AGAIN — two prompts for one payment.
    ("Gowtham — rent 650, must prompt once (2:16)", "dm",
     [("A", "I need 650$ for my rent this month , can you help out a friend?"),
      ("G", "Sure. Will post it now"),
      ("G", "Sending asap"),
      ("G", "Got intent for sending asap"),
      ("A", "Yeah got that"),
      ("G", "Incorrectly got intent for 'got intent for sending asap' message as well"),
      ("G", "Did you too receive an intent"),
      ("A", "No"),
      ("G", "Okay. Cool")],
     [("exactly one money prompt for one payment", lambda p: len(money(p)) == 1),
      ("the prompt carries 650",
       lambda p: not money(p) or "650" in amt(money(p)[0]))]),

    # ── Gowtham, 2:27. Rider asks London -> Windsor; the BOOKER proposes the Toronto
    #    airport instead and the rider never agrees.
    ("Gowtham — London to Windsor, booker suggested Toronto (2:27)", "dm",
     [("G", "We can try"),
      ("A", "Ugh the Ubers surge prices are crazy , maybe I gotta walk to stores from now on"),
      ("G", "Or I can book for you"),
      ("A", "Or maybe you can book me one now to Windsor , Ontario , I'm at London Ontario"),
      ("G", "Can book from your location to the Toronto airport"),
      ("G", "Okay. Booking"),
      ("A", "Did you get ?")],
     [("a ride prompt appears", lambda p: bool(rides(p))),
      ("destination is Windsor — the rider never agreed to Toronto",
       lambda p: not rides(p) or any("windsor" in dest(t) for t in rides(p))),
      ("no prompt sends them to the Toronto airport",
       lambda p: all("toronto" not in dest(t) for t in rides(p))),
      ("pickup is London", lambda p: not rides(p) or any("london" in pick(t) for t in rides(p)))]),

    # ── Yash, 2:20. The offer flow — B never asked, A offers to pay.
    ("Yash — offer 100 rupees, accepted (2:20)", "dm",
     [("A", "Hi"), ("Y", "Hi Akash"),
      ("A", "can i pay you 100 rupees"),
      ("A", "??"),
      ("Y", "Sure"),
      ("A", "Sending now")],
     [("the accepted offer produces a money prompt", lambda p: bool(money(p))),
      ("it carries 100", lambda p: not money(p) or any("100" in amt(t) for t in money(p))),
      ("one prompt, not one per message", lambda p: len(money(p)) <= 1)]),

    # ── Brahma, 2026-08-10. Multi-line address and the stale-trip carryover.
    ("Brahma — cab booking thread (2026-08-10)", "dm",
     [("G", "Hi Brahma"), ("B", "Hello sir"),
      ("G", "Pls book a cab from my location to terminal 2"), ("B", "Sure"),
      ("G", "My address is vakil garden city, Kanakapura road, talaghattapura"), ("B", "Sure"),
      ("G", "Pls book from that to the airport terminal 2"), ("B", "Sure"),
      ("G", "Can you book a can for me\nVakil garden city, Kanakapura road Bangalore\n"
            "To Bengaluru international airport terminal 2"), ("B", "Sure"),
      ("G", "Can you book a cab for me\nVakil garden city, Kanakapura road Bangalore\n"
            "To Bengaluru international airport terminal 2"), ("B", "Sure"),
      ("G", "Book a can from jp nagar to Banashankari"), ("B", "Sure"),
      ("G", "Book a cab from jp nagar to Banashankari"), ("B", "Sure")],
     [("the multi-line address keeps the street",
       lambda p: any("vakil" in pick(t) for t in rides(p))),
      ("the jp nagar trip is its own trip, not the airport one",
       lambda p: any("jp" in pick(t) and "banashankari" in dest(t) for t in rides(p)))]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8002/detect")
    ap.add_argument("--quiet", action="store_true", help="checks only, no transcript")
    a = ap.parse_args()
    bad = 0
    for name, kind, convo, checks in SHOTS:
        rec = play(a.url, kind, convo)
        p = [t for t in rec if t["fired"]]
        print(f"\n=== {name} ===")
        if not a.quiet:
            for t in rec:
                mark = "PROMPT" if t["fired"] else "      "
                print(f"  {mark} {t['who']:2} {t['text'][:56].replace(chr(10), ' / '):58}")
                if t["fired"]:
                    s = t["slots"]
                    if "ride" in t["fired"]:
                        print(f"           -> pickup={s.get('pickup')!r} destination={s.get('destination')!r}")
                    else:
                        print(f"           -> amount={s.get('amount')!r}")
        print(f"  {len(p)} prompt(s)")
        for label, fn in checks:
            try:
                ok = fn(p)
            except Exception as e:
                ok, label = False, f"{label}  [raised {e!r}]"
            bad += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"\n  {'ALL CHECKS PASS' if not bad else f'{bad} CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
