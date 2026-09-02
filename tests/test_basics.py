"""The things that must never be wrong. This is what "no simple issues" means.

Every case here is something the team would call obvious if it failed in front of a
user. It is deliberately not clever - no edge cases, no adversarial phrasings, nothing
that needs a ruling. If a model cannot pass this, it does not ship, regardless of what
any aggregate score says.

Two rules learned the hard way on 2026-09-01:

  Nothing is scored in an empty room. Four separate "model gaps" this repo chased in one
  day turned out to be artifacts of sending a phrase with no conversation in front of it.
  "I'll be sending you that dollar" scores 0.077 alone and 0.996 in its real context.
  Every case below is a conversation.

  Each failure names the layer. A request that never registers is the BASE model (it
  gates request recording at app.py:3344). A response that does not fire is the CONV
  classifier. They need different fixes and conflating them wasted six retrains.

Run against a candidate before shipping it:
    python tests/test_basics.py --url http://127.0.0.1:8900/detect
"""
import argparse, sys, io, time
from collections import defaultdict
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# (category, [(speaker, text), ...], intent, should_fire_on_last)
CASES = [
    # ---------- money: a request, then an acceptance ----------
    ("money request", [("B", "can you send me 500"), ("A", "sure")], "money", True),
    ("money request", [("B", "send me 500 please"), ("A", "ok")], "money", True),
    ("money request", [("B", "pls send 500"), ("A", "yeah")], "money", True),
    ("money request", [("B", "can you gpay me 500"), ("A", "sure")], "money", True),
    ("money request", [("B", "transfer me 500"), ("A", "ok doing it")], "money", True),
    ("money request", [("B", "spot me 500 till friday"), ("A", "sure")], "money", True),
    ("money request", [("B", "lend me 500"), ("A", "ok")], "money", True),
    ("money request", [("B", "need 500 for the cab"), ("A", "sending")], "money", True),
    ("money request", [("B", "you owe me 500"), ("A", "ok will send")], "money", True),
    ("money request", [("B", "can i get 500 from you"), ("A", "sure")], "money", True),

    # ---------- currency forms ----------
    ("currency", [("B", "can you send me $20"), ("A", "sure")], "money", True),
    ("currency", [("B", "can you send me 20 dollars"), ("A", "sure")], "money", True),
    ("currency", [("B", "can you send me rs 500"), ("A", "sure")], "money", True),
    ("currency", [("B", "can you send me 500 rupees"), ("A", "sure")], "money", True),
    ("currency", [("B", "can you send me ₹500"), ("A", "sure")], "money", True),
    ("currency", [("B", "send me 20 bucks"), ("A", "ok")], "money", True),

    # ---------- ride: a request, then an acceptance ----------
    ("ride request", [("B", "can you book me a cab"), ("A", "sure")], "ride", True),
    ("ride request", [("B", "book me a cab to the airport"), ("A", "ok")], "ride", True),
    ("ride request", [("B", "can you get me an uber"), ("A", "sure")], "ride", True),
    ("ride request", [("B", "book an ola for me"), ("A", "yeah")], "ride", True),
    ("ride request", [("B", "need a cab to the office"), ("A", "booking now")], "ride", True),
    ("ride request", [("B", "can you call a cab for me"), ("A", "sure")], "ride", True),
    ("ride request", [("B", "pls book a cab from hsr to koramangala"), ("A", "ok")], "ride", True),
    ("ride request", [("B", "can you book me a ride to the station"), ("A", "sure")], "ride", True),

    # ---------- acceptance vocabulary ----------
    ("acceptance word", [("B", "can you send me 500"), ("A", "yes")], "money", True),
    ("acceptance word", [("B", "can you send me 500"), ("A", "yep")], "money", True),
    ("acceptance word", [("B", "can you send me 500"), ("A", "okay")], "money", True),
    ("acceptance word", [("B", "can you send me 500"), ("A", "cool")], "money", True),
    ("acceptance word", [("B", "can you send me 500"), ("A", "on it")], "money", True),
    ("acceptance word", [("B", "can you send me 500"), ("A", "done deal")], "money", True),
    ("acceptance word", [("B", "can you send me 500"), ("A", "\U0001F44D")], "money", True),
    ("acceptance word", [("B", "can you book me a cab"), ("A", "sure thing")], "ride", True),

    # ---------- self-initiated: acting now, nobody asked ----------
    ("self-initiated", [("A", "hey"), ("A", "sending you 500 now")], "money", True),
    ("self-initiated", [("A", "hey"), ("A", "transferring the 500 now")], "money", True),
    ("self-initiated", [("A", "hey"), ("A", "booking your cab now")], "ride", True),
    ("self-initiated", [("A", "hey"), ("A", "im booking an uber for you")], "ride", True),

    # ---------- offers: the offer waits, the acceptance fires ----------
    ("offer waits", [("A", "shall i transfer it now?")], "money", False),
    ("offer waits", [("A", "want me to send you the money?")], "money", False),
    ("offer waits", [("A", "should i book you a cab?")], "ride", False),
    ("offer waits", [("A", "let me book you a cab")], "ride", False),
    ("offer accepted", [("A", "shall i transfer it now?"), ("B", "yes please")], "money", True),
    ("offer accepted", [("A", "want me to send you the money?"), ("B", "yeah do it")], "money", True),
    ("offer accepted", [("A", "should i book you a cab?"), ("B", "yes please")], "ride", True),
    ("offer accepted", [("A", "let me book you a cab"), ("B", "ok please")], "ride", True),

    # ---------- declines and deferrals must stay quiet ----------
    ("decline", [("B", "can you send me 500"), ("A", "no sorry")], "money", False),
    ("decline", [("B", "can you send me 500"), ("A", "cant right now")], "money", False),
    ("decline", [("B", "can you book me a cab"), ("A", "nah")], "ride", False),
    ("decline", [("A", "shall i transfer it now?"), ("B", "no im good")], "money", False),
    ("deferral", [("B", "can you send me 500"), ("A", "next week")], "money", False),
    ("deferral", [("B", "can you send me 500"), ("A", "ill pay on the 1st")], "money", False),

    # ---------- already done ----------
    ("already done", [("B", "can you send me 500"), ("A", "just sent it")], "money", False),
    ("already done", [("B", "can you book me a cab"), ("A", "cab booked")], "ride", False),
    ("already done", [("B", "did you send it"), ("A", "yes sent yesterday")], "money", False),

    # ---------- questions are not acceptances ----------
    ("question", [("B", "can you send me 500"), ("A", "how much was it again?")], "money", False),
    ("question", [("B", "can you book me a cab"), ("A", "what time?")], "ride", False),

    # ---------- greetings and small talk ----------
    ("greeting", [("B", "can you send me 500"), ("A", "hi")], "money", False),
    ("greeting", [("B", "can you book me a cab"), ("A", "hello")], "ride", False),
    ("small talk", [("B", "can you send me 500"), ("A", "how are you")], "money", False),

    # ---------- other intents must never fire money or ride ----------
    ("not money: food", [("B", "can you order me a pizza"), ("A", "sure")], "money", False),
    ("not ride: food", [("B", "can you order me a pizza"), ("A", "sure")], "ride", False),
    ("not ride: food", [("B", "order some biryani for us"), ("A", "ordering now")], "ride", False),
    ("not money: bill", [("B", "can you pay the electricity bill"), ("A", "sure")], "money", False),
    ("not ride: table", [("B", "can you book a table for 4"), ("A", "sure")], "ride", False),
    ("not ride: ticket", [("B", "can you book the movie tickets"), ("A", "ok")], "ride", False),
    ("not ride: hotel", [("B", "can you book a hotel for us"), ("A", "sure")], "ride", False),
    ("not ride: own car", [("B", "can you drop me at the airport"), ("A", "sure")], "ride", False),
    ("not money: remind", [("B", "remind me to pay him tomorrow"), ("A", "ok")], "money", False),

    # ---------- talking about the product is not using it ----------
    ("meta", [("A", "the payment intent is not showing"), ("B", "yes")], "money", False),
    ("meta", [("A", "i think thats the bug"), ("B", "yes")], "money", False),
    ("meta", [("A", "did the ride intent fire for you?"), ("B", "yes")], "ride", False),
    ("meta", [("A", "will send the new apk now"), ("B", "ok")], "money", False),

    # ---------- past and hypothetical ----------
    ("past tense", [("A", "i booked a cab yesterday"), ("B", "nice")], "ride", False),
    ("past tense", [("A", "i sent him 500 last week"), ("B", "ok")], "money", False),
    ("hypothetical", [("A", "we should book a cab next time"), ("B", "yeah")], "ride", False),
    ("hypothetical", [("A", "what would a cab cost from here"), ("B", "maybe 300")], "ride", False),

    # ---------- a second cycle in the same room ----------
    ("second cycle", [("B", "can you send me 500"), ("A", "sure"),
                      ("B", "can you send me 200 more"), ("A", "ok")], "money", True),
    ("second cycle", [("B", "can you book me a cab"), ("A", "sure"),
                      ("B", "can you book one more"), ("A", "sure")], "ride", True),

    # ---------- an immediate repeat is one prompt, not two ----------
    ("echo", [("B", "can you send me 500"), ("A", "sure"), ("A", "sure")], "money", False),
    ("echo", [("B", "can you book me a cab"), ("A", "ok"), ("A", "ok")], "ride", False),

    # ---------- typos people actually make ----------
    ("typo", [("B", "can you book a can for me"), ("A", "sure")], "ride", True),
    ("typo", [("B", "can you snd me 500"), ("A", "sure")], "money", True),
    ("typo", [("B", "can you send me 50o"), ("A", "sure")], "money", True),
]


# Neutral conversation placed in front of every case.
#
# The classifier reads a ten-message window. A two-turn room fills it with the request
# and the reply and nothing else, which makes both look far more like the whole subject
# of the conversation than they ever are in practice - real rooms run to a median of 25
# messages. Measured on 2026-09-01: "can you send me the slides" / "sending now" scores
# 0.997 in a two-turn room and 0.503 with four messages of ordinary chat in front of it.
# One is a false fire and the other is correct behaviour, and only the padded one is real.
#
# Deliberately about nothing - no amounts, no travel, no plans - so it changes the window
# length without adding signal. Asserted below to fire nothing on its own.
PAD = [
    ("A", "hey"),
    ("B", "hey whats up"),
    ("A", "not much just got back"),
    ("B", "nice how was it"),
    ("A", "pretty good tbh, long day though"),
    ("B", "same here honestly"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8900/detect")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--nopad", action="store_true",
                    help="run without the neutral lead-in, to see which cases are "
                         "context-sensitive. The padded run is the real one.")
    a = ap.parse_args()
    tag = int(time.time())
    pad = [] if a.nopad else PAD

    # the lead-in must not fire on its own, or every result below is contaminated
    if pad:
        room = f"padchk_{tag}"
        for pi, (spk, txt) in enumerate(pad):
            d = requests.post(a.url, timeout=60, json={
                "text": txt, "room_id": room, "sender": spk,
                "message_id": f"{tag}_pad_{pi}"}).json()
            got = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
            assert not got, f"the neutral lead-in fired {got} on {txt!r}"
        print(f"  lead-in: {len(pad)} neutral messages, fires nothing\n")

    bycat = defaultdict(lambda: [0, 0])
    fails = []
    for ci, (cat, turns, intent, want) in enumerate(CASES):
        room = f"basic_{tag}_{ci}"
        base_req = 0.0
        last = None
        for pi, (spk, txt) in enumerate(pad):
            requests.post(a.url, timeout=60, json={
                "text": txt, "room_id": room, "sender": spk,
                "message_id": f"{tag}_{ci}_p{pi}"})
        for ti, (spk, txt) in enumerate(turns):
            try:
                d = requests.post(a.url, timeout=60, json={
                    "text": txt, "room_id": room, "sender": spk,
                    "message_id": f"{tag}_{ci}_{ti}"}).json()
            except Exception as e:
                d = {"_err": repr(e)}
            if ti == 0:
                base_req = float((d.get("scores") or {}).get(intent, 0) or 0)
            last = d
        got = intent in [x for x in (last.get("intents") or []) if x in ("money", "ride")]
        conv = float(((last.get("conversation_state") or {}).get("scores") or {})
                     .get(intent, 0) or 0)
        ok = got == want
        bycat[cat][0] += ok
        bycat[cat][1] += 1
        if not ok:
            # which model is responsible?
            if want and base_req < 0.6 and len(turns) > 1:
                layer = "BASE  (request never recorded)"
            elif want:
                layer = "CONV  (request seen, response did not fire)"
            else:
                layer = "CONV  (fired when it should not)"
            fails.append((cat, turns[-1][1], intent, base_req, conv, layer))
        if a.verbose:
            print(f"  {'ok  ' if ok else 'FAIL'} {cat:18} {turns[-1][1][:38]:38} "
                  f"base={base_req:.3f} conv={conv:.3f}")

    print(f"\n  {'category':22} {'pass':>8}   rate")
    for cat in sorted(bycat, key=lambda c: bycat[c][0] / bycat[c][1]):
        o, n = bycat[cat]
        flag = "   <-- " + "#" * int((1 - o / n) * 10) if o < n else ""
        print(f"  {cat:22} {o:3}/{n:<3} {o/n*100:6.0f}%{flag}")
    tot_o = sum(v[0] for v in bycat.values())
    tot_n = sum(v[1] for v in bycat.values())
    print(f"  {'TOTAL':22} {tot_o:3}/{tot_n:<3} {tot_o/tot_n*100:6.1f}%")

    if fails:
        print(f"\n  --- {len(fails)} failures, by responsible layer ---")
        bylayer = defaultdict(list)
        for cat, txt, intent, b, c, layer in fails:
            bylayer[layer].append((cat, txt, intent, b, c))
        for layer in sorted(bylayer):
            print(f"\n  {layer}  ({len(bylayer[layer])})")
            for cat, txt, intent, b, c in bylayer[layer]:
                print(f"      [{cat}] {intent:5} base={b:.3f} conv={c:.3f}  {txt[:46]!r}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
