"""The base model on its own: does a request register?

The base model (saved_model/, v25 DualHeadRoberta) has exactly one job in the money/ride
path - score a message so that app.py:3344 records a request:

    thr = model_state["thresholds"].get(intent, 0.5)     # money 0.606, ride 0.694
    if result["scores"].get(intent, 0) >= thr:
        request_meta.record(...)

If that gate does not open, nothing is stored, and no later "sure" can fire no matter
what the conversation classifier thinks. So base failures and conv failures are separate
problems with separate fixes, and every retrain so far has gone to the conv model.

This tests the gate directly - one message, one score - because that is exactly how the
gate sees it. That is the ONE place a single message is the right unit; everywhere else
in this repo, scoring a phrase without its conversation produces artifacts.

Two kinds of failure matter:

  MISS   a real request scoring below the gate. Nothing is recorded, so the whole
         conversation is lost. This is the expensive one.
  NOISE  a non-request scoring above it. A spurious request is recorded, which does not
         fire on its own but leaves a live request for an unrelated later "ok" to
         answer. Today the conv classifier masks most of these, which is why they are
         invisible in the fire numbers and still worth knowing about.

    python tests/test_base_model.py --url http://127.0.0.1:8900/detect
"""
import argparse, sys, io, time
from collections import defaultdict
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# (category, text, intent, should_open_a_request)
CASES = [
    # ---------- money requests: must register ----------
    ("money ask", "can you send me 500", "money", True),
    ("money ask", "send me 500", "money", True),
    ("money ask", "pls send 500", "money", True),
    ("money ask", "could you transfer me 500", "money", True),
    ("money ask", "can you gpay me 500", "money", True),
    ("money ask", "can you upi me 500", "money", True),
    ("money ask", "venmo me 20", "money", True),
    ("money ask", "spot me 20 till friday", "money", True),
    ("money ask", "lend me 500", "money", True),
    ("money ask", "can i get 500 from you", "money", True),
    ("money ask", "i need 500 for the cab", "money", True),
    ("money ask", "you owe me 500", "money", True),
    ("money ask", "send the 500 when you can", "money", True),
    ("money ask", "please send it as soon as possible", "money", True),
    ("money ask", "can you just send 300 so i can close this out", "money", True),
    ("money ask", "shoot me the 20", "money", True),
    ("money ask", "wire me the 200", "money", True),

    # ---------- money without a figure ----------
    ("no amount", "can you send me the money", "money", True),
    ("no amount", "send me the cash", "money", True),
    ("no amount", "can you pay me back", "money", True),

    # ---------- currency forms ----------
    ("currency", "can you send me $20", "money", True),
    ("currency", "can you send me 20 dollars", "money", True),
    ("currency", "can you send me rs 500", "money", True),
    ("currency", "can you send me rs.500", "money", True),
    ("currency", "can you send me 500 rupees", "money", True),
    ("currency", "can you send me ₹500", "money", True),
    ("currency", "can you send me 20 bucks", "money", True),
    ("currency", "can you send me 1.5k", "money", True),

    # ---------- ride requests: must register ----------
    ("ride ask", "can you book me a cab", "ride", True),
    ("ride ask", "book me a cab", "ride", True),
    ("ride ask", "can you book a cab for me", "ride", True),
    ("ride ask", "book me a cab to the airport", "ride", True),
    ("ride ask", "can you get me an uber", "ride", True),
    ("ride ask", "book an ola for me", "ride", True),
    ("ride ask", "can you call a cab for me", "ride", True),
    ("ride ask", "i need a cab to the office", "ride", True),
    ("ride ask", "can u grab me a taxi", "ride", True),
    ("ride ask", "arrange a cab for me pls", "ride", True),
    ("ride ask", "can you book me a ride to hsr", "ride", True),
    ("ride ask", "get me a cab in 10 mins", "ride", True),
    ("ride ask", "cab to the station pls", "ride", True),
    ("ride ask", "book a cab from hsr to koramangala", "ride", True),
    ("ride ask", "can you book a rapido for me", "ride", True),

    # ---------- typos people actually make ----------
    ("typo", "can you book a can for me", "ride", True),
    ("typo", "book a can from jp nagar to banashankari", "ride", True),
    ("typo", "can you snd me 500", "money", True),
    ("typo", "can you sedn me 500", "money", True),
    ("typo", "book me a cba", "ride", True),

    # ---------- must NOT open a money request ----------
    ("not money", "can you order me a pizza", "money", False),
    ("not money", "can you book a table for 4", "money", False),
    ("not money", "remind me to pay him tomorrow", "money", False),
    ("not money", "will send the new apk now", "money", False),
    ("not money", "i sent him 500 last week", "money", False),
    ("not money", "how much did you pay for it", "money", False),
    ("not money", "they can't even pay their employees", "money", False),
    ("not money", "send me the utility bill pdf", "money", False),
    ("not money", "send me your address", "money", False),
    ("not money", "send me the link", "money", False),
    ("not money", "what did the cab cost", "money", False),

    # ---------- must NOT open a ride request ----------
    ("not ride", "i booked a cab yesterday", "ride", False),
    ("not ride", "the cab from the airport was late", "ride", False),
    ("not ride", "we should book a cab next time", "ride", False),
    ("not ride", "what would a cab cost from here", "ride", False),
    ("not ride", "can you drop me at the airport", "ride", False),
    ("not ride", "can you book the movie tickets", "ride", False),
    ("not ride", "can you book a hotel for us", "ride", False),
    ("not ride", "heading to office by cab", "ride", False),
    ("not ride", "we'll drop you off by cab and come", "ride", False),
    ("not ride", "did the ride intent fire for you", "ride", False),
    ("not ride", "i can cover the ride", "ride", False),

    # ---------- acceptances are not requests ----------
    ("not a request", "sure", "money", False),
    ("not a request", "yes please send", "money", False),
    ("not a request", "ok will send", "money", False),
    ("not a request", "sending now", "money", False),
    ("not a request", "sure", "ride", False),
    ("not a request", "booking now", "ride", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8900/detect")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    tag = int(time.time())

    # read the live thresholds rather than hardcoding them
    base = a.url.rsplit("/", 1)[0]
    try:
        thr = requests.get(f"{base}/health", timeout=20).json().get("thresholds", {})
    except Exception:
        thr = {}
    TM, TR = float(thr.get("money", 0.606)), float(thr.get("ride", 0.694))
    print(f"  base thresholds in use: money={TM:.3f}  ride={TR:.3f}\n")

    bycat = defaultdict(lambda: [0, 0])
    miss, noise = [], []
    for ci, (cat, text, intent, want) in enumerate(CASES):
        try:
            d = requests.post(a.url, timeout=60, json={
                "text": text, "room_id": f"bm_{tag}_{ci}", "sender": "B",
                "message_id": f"{tag}_{ci}"}).json()
        except Exception:
            d = {}
        score = float((d.get("scores") or {}).get(intent, 0) or 0)
        t = TM if intent == "money" else TR
        opened = score >= t
        ok = opened == want
        bycat[cat][0] += ok
        bycat[cat][1] += 1
        if not ok:
            (miss if want else noise).append((cat, text, intent, score, t))
        if a.verbose:
            print(f"  {'ok  ' if ok else 'FAIL'} {cat:14} {intent:5} {score:.3f}/{t:.3f}  {text[:44]}")

    print(f"  {'category':16} {'pass':>8}   rate")
    for cat in sorted(bycat, key=lambda c: bycat[c][0] / bycat[c][1]):
        o, n = bycat[cat]
        flag = "   <-- " + "#" * int((1 - o / n) * 10) if o < n else ""
        print(f"  {cat:16} {o:3}/{n:<3} {o/n*100:6.0f}%{flag}")
    to = sum(v[0] for v in bycat.values()); tn = sum(v[1] for v in bycat.values())
    print(f"  {'TOTAL':16} {to:3}/{tn:<3} {to/tn*100:6.1f}%")

    if miss:
        print(f"\n  MISS - a real request that never registers ({len(miss)}).")
        print(f"  The conversation is lost regardless of the conv model.")
        for cat, txt, i, s, t in sorted(miss, key=lambda r: -r[3]):
            print(f"    {i:5} {s:.3f} < {t:.3f}  [{cat}] {txt!r}")
    if noise:
        print(f"\n  NOISE - a non-request that opens one anyway ({len(noise)}).")
        print(f"  Leaves a live request for an unrelated later 'ok' to answer.")
        for cat, txt, i, s, t in sorted(noise, key=lambda r: -r[3]):
            print(f"    {i:5} {s:.3f} >= {t:.3f} [{cat}] {txt!r}")
    return 1 if (miss or noise) else 0


if __name__ == "__main__":
    raise SystemExit(main())
