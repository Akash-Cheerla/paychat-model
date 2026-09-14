"""A prompt must carry the request it answers - not an older one, not a newer one.

Bugs from dogfood 2026-09-11, in RequestMeta.take():
  1. A corrected total ("the trip cost was $1000, everyone please send", then "$700")
     and a restated commitment. The second prompt walked past the $700 request its
     sender had already answered and showed $1000: "it turned 100 into 1000".
  2. A reply to an older cab request came back with the newest request's route:
     "Correct for the wrong one, I had replied to Yash's message".

Checks are about ATTRIBUTION, not firing. A case whose message does not fire is
reported as SKIP, not PASS - there are no slots to check on a prompt that never opened.

    python tests/test_stale_slots.py --url http://127.0.0.1:8930/detect
"""
import argparse, io, sys, time

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def unit_checks(results):
    import os
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    os.environ.setdefault("PAYCHAT_CONV_CLASSIFIER", "1")
    from app import RequestMeta, _RESTATE_WINDOW_MSGS

    def room(msgs):
        """msgs: ("req"|"take"|"chat", sender, text). A take passes its text as the
        committing message. Returns the request text each take got."""
        m, got = RequestMeta(), []
        for n, (kind, who, text) in enumerate(msgs, 1):
            m.tick("r")
            if kind == "req":
                amt = next((w for w in text.split() if w.isdigit()), None)
                m.record("r", "money", who, text, f"m{n}", {"amount": f"${amt}"} if amt else {})
            elif kind == "take":
                e, _ = m.take("r", "money", who, text=text)
                got.append(e["text"] if e else None)
        return m, got

    def check(name, ok, detail):
        results.append(("PASS" if ok else "FAIL", name, detail))

    _, got = room([("req", "20", "300 tickets"), ("req", "10", "500 groceries"),
                   ("take", "4", "sure sending"), ("take", "13", "i'll send it"),
                   ("take", "4", "okay sending now")])
    check("unit: restatement keeps its own request when someone else answered another",
          got == ["500 groceries", "300 tickets", "500 groceries"], f"takes {got}")

    _, got = room([("req", "20", "1000 trip"), ("req", "20", "700 trip"),
                   ("take", "4", "okay"), ("req", "4", "so I'll send that much"),
                   ("take", "4", "okay sending that amount now")])
    check("unit: a restatement never walks back to the older $1000 request",
          got == ["700 trip", "700 trip"], f"takes {got}")

    m, _ = room([("req", "20", "1000 trip"), ("req", "20", "700 trip"),
                 ("take", "4", "okay"), ("take", "4", "okay sending that amount now")])
    left = [e["text"] for e in m.rooms["r"]]
    check("unit: a restatement still consumes what the old scan took (firing unchanged)",
          left == [], f"queue {left}")

    _, got = room([("req", "20", "500 bill"), ("take", "4", "sure"),
                   ("req", "20", "200 lunch"), ("take", "4", "ok sending that too")])
    check("unit: a newer open request beats the restatement",
          got == ["500 bill", "200 lunch"], f"takes {got}")

    _, got = room([("req", "20", "300 tickets"), ("take", "4", "sure"),
                   ("req", "10", "500 groceries"), ("take", "4", "ok sent the 300 now")])
    check("unit: stating the answered amount is a restatement even with a newer request open",
          got == ["300 tickets", "300 tickets"], f"takes {got}")

    _, got = room([("req", "20", "60 tickets"), ("req", "20", "40 lunch"),
                   ("take", "4", "sending the 40 now"),
                   ("take", "4", "and sending the 60 too")])
    check("unit: two asks from one person, the stated amount picks each",
          got == ["40 lunch", "60 tickets"], f"takes {got}")

    _, got = room([("req", "20", "60 tickets"), ("req", "20", "40 lunch"),
                   ("take", "4", "sending the 40 now"), ("take", "4", "and the other one too")])
    check("unit: 'the other one' takes the requester's other ask",
          got == ["40 lunch", "60 tickets"], f"takes {got}")

    # continues_for: reusing the previous prompt's figures needs the same request, or a
    # money counter-offer made by the person firing now
    def carries(steps, final_sender, intent="money"):
        m = RequestMeta()
        last = None
        for kind, who, text, slots in steps:
            m.tick("r")
            if kind == "req":
                m.record("r", intent, who, text, text, slots, divisible=(intent == "ride"))
            else:
                e, _ = m.take("r", intent, who, text=text)
                m.remember_resolved("r", intent, slots, src=e)
                last = e
        e, _ = m.take("r", intent, final_sender)
        return (e or {}).get("text"), m.continues_for("r", intent, e, final_sender)

    got = carries([("req", "4", "home to shivaji nagar", {}), ("req", "20", "yash home to jp nagar", {}),
                   ("take", "20", "sure let me book", {"destination": "Shivaji Nagar"})], "10", "ride")
    check("unit: a ride prompt for a newer request does not inherit the last route",
          got == ("yash home to jp nagar", False), f"{got}")
    got = carries([("req", "A", "spot me 5000", {"amount": "$5000"}), ("req", "B", "how about 3000", {"amount": "$3000"}),
                   ("take", "A", "yeah 3000 helps", {"amount": "$3000"})], "B")
    check("unit: a counter-offer still carries 3000 to the payer's next prompt",
          got == ("spot me 5000", True), f"{got}")
    got = carries([("req", "A", "25 coffee", {"amount": "$25"}), ("req", "B", "80 gas", {"amount": "$80"}),
                   ("take", "C", "sending", {"amount": "$80"})], "D")
    check("unit: two requesters - the second payer does not inherit $80",
          got == ("25 coffee", False), f"{got}")

    _, got = room([("req", "20", "700 trip"), ("take", "4", "okay")]
                  + [("chat", "20", "lol")] * _RESTATE_WINDOW_MSGS
                  + [("take", "4", "okay sending now")])
    check(f"unit: past {_RESTATE_WINDOW_MSGS} messages a restatement gets nothing",
          got == ["700 trip", None], f"takes {got}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8930/detect")
    a = ap.parse_args()
    tag = int(time.time())
    results = []

    def send(room, n, sender, text, reply_to=None, participants=None):
        body = {"text": text, "room_id": room, "sender": sender, "message_id": f"{room}_{n}"}
        if reply_to:
            body["reply_to"] = reply_to
        if participants:
            body["participants"] = participants
        return requests.post(a.url, json=body, timeout=90).json()

    def tb(d):
        return (d.get("conversation_state") or {}).get("triggered_by") or {}

    def check(name, d, intent, ok, detail):
        if intent not in (d.get("intents") or []):
            results.append(("SKIP", name, f"{intent} did not fire"))
        else:
            results.append(("PASS" if ok(d) else "FAIL", name, detail(d)))

    def show(d):
        t = tb(d)
        s = t.get("slots") or {}
        keep = {k: v for k, v in s.items() if k in ("amount", "pickup", "destination") and v}
        return f"matched {str(t.get('text'))[:50]!r} slots {keep}"

    def amount(d):
        return str((tb(d).get("slots") or {}).get("amount") or (d.get("slots") or {}).get("amount")
                   or (d.get("money") or {}).get("detected_amount") or "")

    # 1. corrected total, then the same person restates their commitment. "everyone please
    #    send" on a total is a split: blank when the group size is unknown (Gowtham,
    #    2026-09-13), the share when the backend sends participants.
    for parts, want in ((None, ""), (7, "100")):
        r = f"stale_money_{parts}_{tag}"
        send(r, 1, "20", "Hey guys , the trip cost was $1000 , everyone please send me the money", participants=parts)
        send(r, 2, "4", "Actually uber doesn't send ride booking confirmations right after", participants=parts)
        send(r, 3, "20", "Hey guys , the trip cost was $700 , everyone please send me the money", participants=parts)
        d = send(r, 4, "4", "Okay", participants=parts)
        label = f"participants={parts}: " + ("amount blank" if not want else f"${want} share")
        check(f"corrected total, first answer: $700 request, {label}", d, "money",
              lambda d, want=want: "700" in str(tb(d).get("text")) and
              (want in amount(d) if want else amount(d) == ""), show)
        send(r, 5, "4", "Well since it's my share only, it'd be 100 right", participants=parts)
        send(r, 6, "4", "So I'll send that much", participants=parts)
        send(r, 7, "20", "Yes , sure", participants=parts)
        d = send(r, 8, "4", "Okay, sending that amount now", participants=parts)
        check(f"restated: still the $700 request, never $1000, {label}", d, "money",
              lambda d, want=want: "700" in str(tb(d).get("text")) and "1000" not in amount(d)
              and (want in amount(d) if want else amount(d) == ""), show)

    # 2. a reply to the OLDER of two cab requests takes that request's route
    r = f"stale_ride_reply_{tag}"
    send(r, 1, "13", "Can someone help me book a cab from my home to mumbai airport")
    send(r, 2, "20", "Can someone book a cab from my current location to University of Windsor?")
    d = send(r, 3, "4", "Sure, booking it", reply_to=f"{r}_1")
    check("reply_to the older request takes its route", d, "ride",
          lambda d: tb(d).get("message_id") == f"{r}_1", show)

    # 3. control: no reply, the newest request wins
    r = f"stale_ride_recent_{tag}"
    send(r, 1, "13", "Can someone help me book a cab from my home to mumbai airport")
    send(r, 2, "20", "Can someone book a cab from my current location to University of Windsor?")
    d = send(r, 3, "4", "Sure, booking it")
    check("no reply: the newest request wins", d, "ride",
          lambda d: tb(d).get("message_id") == f"{r}_2", show)

    # 4. a reply_to that names nothing falls back to recency instead of failing
    r = f"stale_ride_unknown_{tag}"
    send(r, 1, "13", "Can someone help me book a cab from my home to mumbai airport")
    send(r, 2, "20", "Can someone book a cab from my current location to University of Windsor?")
    d = send(r, 3, "4", "Sure, booking it", reply_to="no_such_message")
    check("unknown reply_to falls back to the newest request", d, "ride",
          lambda d: tb(d).get("message_id") == f"{r}_2", show)

    # 5. a split is still owed by everyone - two different payers both get the request
    r = f"stale_split_{tag}"
    send(r, 1, "20", "Dinner was 900 , thats 300 each , everyone please send")
    d1 = send(r, 2, "4", "Sending now")
    send(r, 3, "10", "The food was great")
    d2 = send(r, 4, "13", "Sending mine too")
    check("split: first payer gets the dinner request", d1, "money",
          lambda d: "900" in str(tb(d).get("text")), show)
    check("split: second payer still gets it", d2, "money",
          lambda d: "900" in str(tb(d).get("text")), show)

    # 6. a NEW request after answering: the next commitment must not reuse the old figure
    r = f"stale_newreq_{tag}"
    send(r, 1, "20", "Can you send me 500 for the electricity bill")
    send(r, 2, "4", "Sure sending now")
    send(r, 3, "20", "Also can you send me 200 for lunch")
    d = send(r, 4, "4", "Ok sending that too")
    check("new request after answering: no $500 on the next sheet", d, "money",
          lambda d: "500" not in str((tb(d).get("slots") or {}).get("amount"))
          and "500" not in str(tb(d).get("text")), show)

    # 7. two separate asks from one person: the stated amount picks the right one
    r = f"stale_twoasks_{tag}"
    send(r, 1, "20", "Can you send me 60 for the movie tickets")
    send(r, 2, "4", "one sec")
    send(r, 3, "20", "Also send me 40 for lunch")
    send(r, 4, "4", "Sending the 40 now")
    for n, (who, txt) in enumerate([("20", "thanks"), ("4", "np"), ("20", "cool"),
                                    ("4", "ok"), ("20", "hmm")], 5):
        send(r, n, who, txt)
    d = send(r, 10, "4", "And sending the 60 for the tickets too")
    check("two asks from one person: 'sending the 60 too' gets the $60 request", d, "money",
          lambda d: "60" in str(tb(d).get("text"))
          and "40" not in str((tb(d).get("slots") or {}).get("amount")), show)

    # 7b. "the other one" with no amount: the other ask, amount blank (Gowtham, 2026-09-13)
    r = f"stale_otherone_{tag}"
    send(r, 1, "20", "Can you send me 60 for the movie tickets")
    send(r, 2, "20", "Also send me 40 for lunch")
    send(r, 3, "4", "Sending the 40 now")
    for n, (who, txt) in enumerate([("P", "lol"), ("Q", "ok"), ("P", "nice"), ("Q", "haha"),
                                    ("P", "k")], 4):
        send(r, n, who, txt)
    d = send(r, 9, "4", "And sending the other one too")
    check("'the other one': the $60 request, amount blank", d, "money",
          lambda d: "60" in str(tb(d).get("text")) and amount(d) == "", show)

    # 8. a partial payment types its own amount; the sheet shows it, not the request's
    r = f"dm_stale_partial_{tag}"
    msgs = [("A", "can you send me 700"), ("B", "okay"), ("A", "cool"), ("B", "hmm"),
            ("A", "?"), ("B", "ya"), ("A", "k")]
    for n, (who, txt) in enumerate(msgs, 1):
        send(r, n, who, txt)
    d = send(r, 8, "B", "sending you 100 now, rest later")
    check("DM partial payment: sheet shows the typed $100, not $700", d, "money",
          lambda d: "100" in str((tb(d).get("slots") or {}).get("amount") or (d.get("slots") or {}).get("amount"))
          and "700" not in str((d.get("slots") or {}).get("amount")), show)

    # 9. group: a share typed after the corrected total
    r = f"stale_share_{tag}"
    msgs = [("20", "Hey guys, the trip cost was $1000, everyone please send me the money"),
            ("4", "hmm"), ("20", "Hey guys, the trip cost was $700, everyone please send me the money"),
            ("4", "Okay"), ("20", "thanks"), ("P", "lol"), ("Q", "ok"), ("P", "haha")]
    for n, (who, txt) in enumerate(msgs, 1):
        send(r, n, who, txt)
    d = send(r, 9, "4", "Sending you 100 now")
    check("group share typed after a correction: $100, request is the $700 one", d, "money",
          lambda d: "100" in str((tb(d).get("slots") or {}).get("amount"))
          and "700" in str(tb(d).get("text")), show)

    # 8+. RequestMeta bookkeeping, checked directly. Over HTTP these depend on the
    # classifier firing at an exact message distance, which it often does not.
    unit_checks(results)

    for status, name, detail in results:
        print(f"  {status:4}  {name:52}  {detail}")
    scored = [x for x in results if x[0] != "SKIP"]
    passed = sum(1 for x in scored if x[0] == "PASS")
    print(f"\n  TOTAL {passed}/{len(scored)}   ({len(results) - len(scored)} skipped - did not fire)")
    return 0 if passed == len(scored) else 1


if __name__ == "__main__":
    raise SystemExit(main())
