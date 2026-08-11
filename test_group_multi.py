"""Group splits, multi-payer, and one message carrying two intents.

The paths here are the ones with the most ways to be quietly wrong: a split can divide
by the wrong number, a group request can be answered by the wrong person, and a single
message can open a payment and a booking at once. None of these show up in a DM.

`participants` is the headcount the backend sends (group rooms only). Several cases run
with and without it, because the whole point of the field is what happens when the
message does not state the count.

Run:  python test_group_multi.py [--url http://127.0.0.1:8002/detect]
"""
import argparse, itertools, sys, time

import requests

_ids = itertools.count(int(time.time() * 1000) % 1_000_000_000)


def play(url, convo, participants=None, dm=False):
    n = next(_ids)
    room = f"dm_{n}_{n + 3}" if dm else f"grp_{n}"
    out = []
    for i, (who, text) in enumerate(convo):
        body = {"text": text, "room_id": room, "sender": who, "message_id": f"m{i}"}
        if participants:
            body["participants"] = participants
        r = requests.post(url, json=body, timeout=180).json()
        cs = r.get("conversation_state") or {}
        out.append({"who": who, "text": text,
                    "fired": [x for x in (r.get("intents") or []) if x in ("money", "ride")],
                    "slots": (cs.get("triggered_by") or {}).get("slots") or {}})
    return [t for t in out if t["fired"]]


def amt(t):
    return str(t["slots"].get("amount") or "")


def dest(t):
    return str(t["slots"].get("destination") or "").lower()


CASES = [
    # ── splits ──
    ("split with a stated count divides by it", None,
     [("A", "dinner was 3000, split 3 ways"), ("B", "sending my part now")],
     lambda p: p and "1000" in amt(p[-1]),
     "1000, not the 3000 total"),

    ("split with no count but participants=3 divides by 3", 3,
     [("A", "dinner came to 3000, send me your shares"), ("B", "sending mine now")],
     lambda p: p and "1000" in amt(p[-1]),
     "1000 — headcount came from the backend"),

    ("split with no count and no participants does NOT guess", None,
     [("A", "dinner came to 3000, send me your shares"), ("B", "sending mine now")],
     lambda p: p and "3000" not in amt(p[-1]),
     "blank rather than the full 3000"),

    ("a stated per-person figure is not divided again", 5,
     [("A", "dinner was 5000, thats 1000 each"), ("B", "sending now")],
     lambda p: p and "1000" in amt(p[-1]),
     "1000, not 200"),

    # ── "lets split it": a split with no count stated anywhere ──
    #
    # _DIVISIBLE only matched a split with an explicit count ("split 3 ways",
    # "between 4 of us"), so the commonest phrasing of all was not treated as a split
    # at all and the FULL total landed on the payment sheet.
    ("'lets split it' with participants divides", 3,
     [("A", "dinner was 3000, lets split it"), ("B", "sending my part now")],
     lambda p: p and "1000" in amt(p[-1]),
     "1000"),

    ("'lets split it' with no headcount does NOT show the total", None,
     [("A", "dinner was 3000, lets split it"), ("B", "sending my part now")],
     lambda p: p and "3000" not in amt(p[-1]),
     "blank, never the full 3000"),

    ("'lets split this' phrasing too", 4,
     [("A", "the tab was 4000, lets split this"), ("B", "sending mine"), ],
     lambda p: p and "1000" in amt(p[-1]),
     "1000"),

    # ── a question ABOUT an amount is not a request FOR it ──
    #
    # Any message whose raw money score crossed threshold was recorded as an open
    # request, and questions about a debt score high ("how much do you owe me?" =
    # 0.787). Each spurious request gave a later commitment something extra to answer,
    # so one $20 debt produced four payment sheets.
    ("asking how much is owed does not open a request", None,
     [("A", "How much do you owe me?"), ("G", "I remember it as twenty bucks"),
      ("A", "Can you send that now please?"), ("G", "Sure"),
      ("G", "Will send the cash now")],
     lambda p: len(p) == 1,
     "one prompt — the question is not a second request to answer"),

    # "sending now" IS a self-initiated payment and fires on its own under §3 — the
    # question before it is not what makes it fire, and must not donate an amount.
    ("asking how much to send opens no request of its own", None,
     [("A", "how much should i send you?"), ("B", "no idea"), ("A", "sending now")],
     lambda p: len(p) <= 1 and not any(amt(t) for t in p),
     "at most one prompt, and no amount lifted from the question"),

    # must not regress: these ARE requests and must still fire
    ("'you owe me 500' still fires", None,
     [("A", "you still owe me 500"), ("B", "sending it now")],
     lambda p: len(p) == 1 and "500" in amt(p[0]),
     "one prompt for 500"),

    ("'can you send me 500?' still fires", None,
     [("A", "can you send me 500?"), ("B", "sure sending")],
     lambda p: len(p) == 1 and "500" in amt(p[0]),
     "one prompt for 500"),

    # ── multi-payer: a split is owed by everyone, not consumed by the first ──
    ("every payer on a split gets their own prompt", 3,
     [("A", "dinner came to 3000, send me your shares"),
      ("B", "sending mine now"), ("C", "sending mine too")],
     lambda p: len(p) == 2 and all("1000" in amt(t) for t in p),
     "two prompts, 1000 each"),

    ("the same payer restating does not get a second prompt", 3,
     [("A", "dinner came to 3000, send me your shares"),
      ("B", "sending mine now"), ("B", "sent it")],
     lambda p: len(p) == 1,
     "one prompt for B"),

    # ── one message, two intents ──
    ("one message opening money AND ride", None,
     [("A", "send me 900 and book me a cab to hsr"), ("B", "ok doing both now")],
     lambda p: {"money", "ride"} <= {i for t in p for i in t["fired"]},
     "both a payment and a booking"),

    ("two open requests, a bare ack is ambiguous", None,
     [("A", "can someone send me 500"), ("B", "also can someone book me a cab to hsr"),
      ("C", "ok")],
     lambda p: len(p) <= 1,
     "at most one prompt — a bare 'ok' cannot answer both (rule 6a)"),

    # ── who answers ──
    ("someone other than the asker can answer a group request", None,
     [("A", "can someone book a cab to the airport"), ("C", "i'll book it")],
     lambda p: len(p) == 1 and "airport" in dest(p[0]),
     "C's prompt carries the airport"),

    ("the asker answering their own request does not fire", None,
     [("A", "can someone send me 500"), ("A", "actually never mind ill manage")],
     lambda p: len(p) == 0,
     "no prompt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8002/detect")
    a = ap.parse_args()
    bad = 0
    for name, participants, convo, check, expect in CASES:
        p = play(a.url, convo, participants=participants)
        try:
            ok = check(p)
        except Exception as e:
            ok, expect = False, f"{expect}  [raised {e!r}]"
        tag = f"participants={participants}" if participants else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name} {tag}")
        if not ok:
            for t in p:
                print(f"          PROMPT {t['who']} {t['fired']} {t['slots']}")
            print(f"          expect {expect}")
        bad += not ok
    print(f"\n  {len(CASES) - bad}/{len(CASES)} passing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
