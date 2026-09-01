"""The list Gowtham asked for, run against the model instead of written down.

"Come up with a list of simple chat message variations of asking/requesting for money or
rides, both rupees and dollars, that you guys think should work 100% cases."

A request does not fire on its own - it fires when somebody agrees to it - so each
variation is sent followed by a plain "sure". If the pair does not produce a prompt, that
phrasing does not work, and a list of phrasings nobody has run is worth very little.

Grouped so the failures point somewhere: if the rupee column fails and the dollar column
passes, that is a currency problem, not a phrasing one.
"""
import argparse, sys, io, time
from collections import defaultdict
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MONEY_DOLLAR = [
    "can you send me 20$", "can you send me 20 dollars", "send me 20$ please",
    "pls send 20 bucks", "gimme 20 bucks", "spot me 20$ till friday",
    "can you spot me 20 dollars", "lend me 20$", "transfer me 20$",
    "can you venmo me 20", "need 20$ for dinner", "you owe me 20$",
    "can i get 20 bucks from you", "20$ for the tickets pls",
]
MONEY_RUPEE = [
    "can you send me 500 rupees", "can you send me ₹500", "send me rs 500",
    "pls send 500 rs", "gimme 500 na", "spot me 500 till salary",
    "can you gpay me 500", "can you upi me 500", "transfer 500 rupees",
    "need 500 for the cab", "you owe me 500 rupees", "500 rs for lunch pls",
    "can i get 500 rupees", "phonepe me 500",
]
RIDE = [
    "can you book me a cab", "can you book a cab for me", "book me a cab to the airport",
    "pls book a cab", "can you get me an uber", "book an ola for me",
    "can you call a cab for me", "need a cab to the office",
    "book a cab from my location to marathalli", "can u grab me a taxi",
    "arrange a cab for me pls", "can you book a ride to hsr",
    "get me a cab in 10 mins", "cab to the station pls",
]

GROUPS = [("money · dollars", MONEY_DOLLAR), ("money · rupees", MONEY_RUPEE),
          ("ride", RIDE)]
INTENT = {"money · dollars": "money", "money · rupees": "money", "ride": "ride"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8900/detect")
    ap.add_argument("--reply", default="sure")
    a = ap.parse_args()
    tag = int(time.time())
    totals = defaultdict(lambda: [0, 0])
    fails = []
    n = 0
    for gname, reqs in GROUPS:
        intent = INTENT[gname]
        print(f"\n  === {gname} — request, then \"{a.reply}\" ===")
        for req in reqs:
            room = f"rv_{tag}_{n}"; n += 1
            requests.post(a.url, timeout=60, json={
                "text": req, "room_id": room, "sender": "B",
                "message_id": f"{tag}_{n}_q"})
            d = requests.post(a.url, timeout=60, json={
                "text": a.reply, "room_id": room, "sender": "A",
                "message_id": f"{tag}_{n}_a"}).json()
            got = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
            sc = (d.get("conversation_state") or {}).get("scores", {}) or {}
            ok = intent in got
            totals[gname][0] += ok
            totals[gname][1] += 1
            if not ok:
                fails.append((gname, req, float(sc.get(intent, 0) or 0)))
            print(f"    {'ok  ' if ok else 'FAIL'}  {sc.get(intent,0):.3f}  {req}")

    print(f"\n  === summary ===")
    tot = [0, 0]
    for g, (o, t) in totals.items():
        tot[0] += o; tot[1] += t
        print(f"    {g:18} {o}/{t}  = {o/t*100:.0f}%")
    print(f"    {'ALL':18} {tot[0]}/{tot[1]}  = {tot[0]/tot[1]*100:.0f}%")
    if fails:
        print(f"\n  phrasings that do NOT work:")
        for g, req, s in fails:
            print(f"    [{g}] {s:.3f}  {req!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
