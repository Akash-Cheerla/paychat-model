"""Robustness suite: does the model understand the RULE, or only the phrasings it was fed?

v14's real-traffic misses were single-word brittleness: "can you give me 10₹" -> "ok"
scored 0.08 while "$10" scored 0.997; "why not", "oh ok", "yep fr" scored ~0.03 where
"ok" scored 0.997. This suite varies ONE factor at a time across many phrasings:

  positives (must fire)   acceptance wording, request wording, amount format, ride wording,
                          agreement + detail question, offers and their acceptance,
                          interruptions between request and acceptance
  negatives (must stay    declines, questions with no agreement, future promises, already
  quiet)                  done, hedges, the requester's own follow-up, "ok"/"why not" with
                          no request, declined offers
  sequences (pipeline)    re-sent request, retry after failure, group answerers, duplicates

Every phrase comes from tests/robustness_lexicon.py and is scored under TRAIN or HELDOUT.
A training generator may only use TRAIN phrasings, so HELDOUT measures generalisation.
Labels follow FIRING_RULE.md with the 2026-09-13 and 2026-09-15 rulings.

    python tests/robustness_suite.py --url http://127.0.0.1:8971/detect --name v14 [--show 40]
"""
import argparse, io, json, random, sys, time
from collections import defaultdict
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robustness_lexicon as L   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = random.Random(2026)


def amt(fmt=None, n=None):
    return (fmt or R.choice(L.split(L.AMOUNTS)[0])).format(n=n or R.choice(L.NUMBERS))


def money_req(tpl=None, fmt=None):
    tpl = tpl or R.choice(L.split(L.MONEY_REQUESTS)[0])
    return tpl.format(amt=amt(fmt))


def ride_req(tpl=None):
    tpl = tpl or R.choice(L.split(L.RIDE_REQUESTS)[0])
    a, b = R.sample(L.PLACES, 2)
    return tpl.format(place=R.choice(L.PLACES), a=a, b=b)


def first_train(items):
    return L.split(items)[0][0]


def build():
    cases = []   # (family, tested_phrases, want, room_kind, msgs)

    def add(family, tested, want, msgs, kind="dm"):
        cases.append({"family": family, "tested": tested, "want": want, "kind": kind, "msgs": msgs})

    simple_accept = first_train(L.ACCEPT)
    for p in L.ACCEPT + L.ACCEPT_MONEY_ACTION:
        for _ in range(2):
            add("P1 money acceptance wording", [p], "money", [("1", money_req()), ("2", p)])
    for p in L.ACCEPT + L.ACCEPT_RIDE_ACTION:
        add("P5 ride acceptance wording", [p], "ride", [("1", ride_req()), ("2", p)])
    for t in L.MONEY_REQUESTS:
        for _ in range(2):
            add("P2 money request wording", [t], "money", [("1", money_req(tpl=t)), ("2", simple_accept)])
    for f in L.AMOUNTS:
        for _ in range(2):
            add("P3 amount format", [f], "money", [("1", money_req(fmt=f)), ("2", simple_accept)])
    for t in L.RIDE_REQUESTS:
        for _ in range(2):
            add("P4 ride request wording", [t], "ride", [("1", ride_req(tpl=t)), ("2", simple_accept)])
    for p in L.ACCEPT_WITH_QUESTION_MONEY:
        for _ in range(2):
            add("P6 agreement + question (ruling 1)", [p], "money", [("1", money_req()), ("2", p)])
    for p in L.ACCEPT_WITH_QUESTION_RIDE:
        for _ in range(2):
            add("P6 agreement + question (ruling 1)", [p], "ride", [("1", ride_req()), ("2", p)])
    acc_train = L.split(L.OFFER_ACCEPT)[0]
    for o in L.OFFERS_MONEY:
        for _ in range(2):
            add("P7 offer, then accepted", [o], "money", [("1", o.format(amt=amt())), ("2", R.choice(acc_train))])
    for o in L.OFFERS_RIDE:
        for _ in range(2):
            add("P7 offer, then accepted", [o], "ride", [("1", o), ("2", R.choice(acc_train))])
    off_train_m = L.split(L.OFFERS_MONEY)[0]
    for a in L.OFFER_ACCEPT:
        add("P7 offer, then accepted", [a], "money", [("1", R.choice(off_train_m).format(amt=amt())), ("2", a)])
    for x in L.INTERRUPT_BY_ACCEPTOR:
        for _ in range(2):
            add("P8 interruption before acceptance", [x], "money", [("1", money_req()), ("2", x), ("2", simple_accept)])
    for x in L.INTERRUPT_BY_REQUESTER:
        for _ in range(2):
            add("P8 interruption before acceptance", [x], "money", [("1", money_req()), ("1", x), ("2", simple_accept)])

    for p in L.DECLINE:
        add("N1 decline", [p], None, [("1", money_req()), ("2", p)])
        add("N1 decline", [p], None, [("1", ride_req()), ("2", p)])
    for p in L.QUESTION_ONLY_MONEY:
        for _ in range(2):
            add("N2 question, no agreement", [p], None, [("1", money_req()), ("2", p)])
    for p in L.QUESTION_ONLY_RIDE:
        for _ in range(2):
            add("N2 question, no agreement", [p], None, [("1", ride_req()), ("2", p)])
    for p in L.FUTURE:
        for _ in range(2):
            add("N3 future promise", [p], None, [("1", money_req() if "book" not in p else ride_req()), ("2", p)])
    for p in L.DONE:
        for _ in range(2):
            add("N4 already done", [p], None, [("1", money_req() if "book" not in p else ride_req()), ("2", p)])
    for p in L.HEDGE:
        for _ in range(2):
            add("N5 hedge", [p], None, [("1", money_req()), ("2", p)])
    for p in L.HARD_NEGATIVE:
        add("N9 hard negative (looks like yes)", [p], None, [("1", money_req()), ("2", p)])
        add("N9 hard negative (looks like yes)", [p], None, [("1", ride_req()), ("2", p)])
    for p in L.REQUESTER_FOLLOWUP:
        for _ in range(2):
            add("N6 requester's own follow-up", [p], None, [("1", money_req()), ("1", p)])
    acks = ["ok", "why not", "oh ok", "sure", "yep fr", "alright"]
    for o in L.NON_REQUEST_OPENERS:
        for a in acks:
            add("N7 acknowledgement, no request", [a], None, [("1", o), ("2", a)])
    for p in L.DECLINE[:8]:
        add("N8 offer declined", [p], None, [("1", R.choice(off_train_m).format(amt=amt())), ("2", p)])

    # ---- added 2026-09-17 (v17b). Appended AFTER the families above so the RNG draws for
    # every earlier case are unchanged and old baselines stay comparable.
    place = lambda: R.choice(L.PLACES)
    group_money = lambda: "can someone " + R.choice(["send me {amt}", "venmo me {amt}", "spot me {amt}",
                                                      "cover {amt} for me"]).format(amt=amt())
    group_ride = lambda: "can someone " + R.choice(["book me a cab to {p}", "get me an uber to {p}",
                                                     "order me an ola to {p}"]).format(p=place())
    pre = lambda: [("2", R.choice(L.NON_REQUEST_OPENERS))] if R.random() < 0.5 else []
    for p in L.SELF_SEND:
        for _ in range(2):
            add("P9 self-initiated send or book", [p], "money", pre() + [("1", p.format(amt=amt()))])
    for p in L.SELF_BOOK:
        for _ in range(2):
            add("P9 self-initiated send or book", [p], "ride", pre() + [("1", p.format(place=place()))])
    for o in L.OFFERS_MONEY:
        add("N10 offer alone", [o], None, [("1", o.format(amt=amt()))])
    for o in L.OFFERS_RIDE:
        add("N10 offer alone", [o], None, [("1", o)])
    debt_train = L.split(L.DEBT_STATEMENTS)[0]
    for d in L.DEBT_STATEMENTS:
        add("P10 debt, reply agrees to pay (ruling 8)", [d], "money", [("1", d.format(amt=amt())), ("2", "sure")])
        add("N11 debt, reply only admits it (ruling 8)", [d], None, [("1", d.format(amt=amt())), ("2", "alright")])
    for p in L.DEBT_PAY_AGREE:
        for _ in range(2):
            add("P10 debt, reply agrees to pay (ruling 8)", [p], "money", [("1", R.choice(debt_train).format(amt=amt())), ("2", p)])
    for p in L.DEBT_ADMIT:
        for _ in range(2):
            add("N11 debt, reply only admits it (ruling 8)", [p], None, [("1", R.choice(debt_train).format(amt=amt())), ("2", p)])
    short_train = L.split(L.SHORTFALL)[0]
    for s in L.SHORTFALL:
        add("N12 shortfall, polite noise", [s], None, [("1", s.format(amt=amt())), ("2", R.choice(["sure", "ok"]))])
    for p in L.SHORTFALL_POLITE:
        add("N12 shortfall, polite noise", [p], None, [("1", R.choice(short_train).format(amt=amt())), ("2", p)])
    for p in L.SHORTFALL_COMMIT:
        for _ in range(2):
            add("P11 shortfall, then a commitment", [p], "money",
                [("1", R.choice(short_train).format(amt=amt())), ("2", p.format(amt=amt()))])
    take_m, take_r = L.split(L.CLEAR_TAKE_MONEY)[0], L.split(L.CLEAR_TAKE_RIDE)[0]
    bare = ["okay", "ok", "sure", "cool", "k", "alright", "nice"]
    for t in L.CLEAR_TAKE_MONEY:
        add("P12 clear take answers a request (s3c)", [t], "money", [("A", group_money()), ("B", t)], kind="group")
        add("N13 3rd person's bare ok after a clear take (ruling 6)", [t], None,
            [("A", group_money()), ("B", t), ("C", R.choice(bare))], kind="group")
    for t in L.CLEAR_TAKE_RIDE:
        add("P12 clear take answers a request (s3c)", [t], "ride", [("A", group_ride()), ("B", t)], kind="group")
        add("N13 3rd person's bare ok after a clear take (ruling 6)", [t], None,
            [("A", group_ride()), ("B", t), ("C", R.choice(bare))], kind="group")
    for p in L.THIRD_EXPLICIT:
        for _ in range(2):
            ride = "book" in p
            add("P13 3rd person's clear words after a clear take", [p], "ride" if ride else "money",
                [("A", group_ride() if ride else group_money()), ("B", R.choice(take_r if ride else take_m)), ("C", p)],
                kind="group")
    for c in ["sure", "ok", "yes", "yeah", "yep", "okay"]:
        add("P14 2nd bare answer after a bare answer (ruling 7)", [c], "money",
            [("A", group_money()), ("B", R.choice(["sure", "ok", "yes"])), ("C", c)], kind="group")
        add("P14 2nd bare answer after a bare answer (ruling 7)", [c], "ride",
            [("A", group_ride()), ("B", R.choice(["sure", "ok", "yes"])), ("C", c)], kind="group")
    for p in L.ACCEPT_MONEY_ACTION:
        add("P15 two open requests (s6a)", [p], "money", [("A", money_req()), ("C", "also " + ride_req()), ("B", p)], kind="group")
        add("P15 two open requests (s6a)", [p], "money", [("C", ride_req()), ("A", "also " + money_req()), ("B", p)], kind="group")
    for p in L.ACCEPT_RIDE_ACTION:
        add("P15 two open requests (s6a)", [p], "ride", [("A", money_req()), ("C", "also " + ride_req()), ("B", p)], kind="group")
        add("P15 two open requests (s6a)", [p], "ride", [("C", ride_req()), ("A", "also " + money_req()), ("B", p)], kind="group")
    for b in ["ok", "sure", "okay"]:
        add("P15 two open requests (s6a)", [b], "ride", [("A", money_req()), ("C", "also " + ride_req()), ("B", b)], kind="group")
        add("P15 two open requests (s6a)", [b], "money", [("C", ride_req()), ("A", "also " + money_req()), ("B", b)], kind="group")

    # Sequences - pipeline behaviour under the rulings; the phrase split does not apply.
    S = "S sequences"
    add(S + ": re-sent same request, 2nd 'sure' quiet (ruling 2)", [], None,
        [("1", "can you send me $20"), ("2", "sure"), ("1", "can you send me $20"), ("2", "sure")])
    add(S + ": re-sent with a new amount fires (ruling 2)", [], "money",
        [("1", "can you send me $20"), ("2", "sure"), ("1", "actually can you send me $50"), ("2", "sure")])
    add(S + ": re-sent after a failure fires (ruling 5)", [], "money",
        [("1", "can you send me $20"), ("2", "sure"), ("2", "did not get an intent"), ("1", "can you send me $20"), ("2", "sure")])
    add(S + ": second group answerer fires (ruling 3)", [], "money",
        [("1", "can someone send me 10 dollars"), ("2", "sure"), ("3", "sure")], kind="group")
    add(S + ": answer after 'sent it' is quiet (ruling 3)", [], None,
        [("1", "can someone send me 10 dollars"), ("2", "sure"), ("2", "sent it"), ("3", "sure")], kind="group")
    add(S + ": split stays open after one 'sent' (ruling 3)", [], "money",
        [("1", "dinner was 900, everyone send 300"), ("2", "sending"), ("2", "sent"), ("3", "sending mine")], kind="group")
    add(S + ": same person restating is quiet", [], None,
        [("1", "can you send me $20"), ("2", "ok"), ("2", "sending now")])
    add(S + ": offerer announcing after acceptance is quiet", [], None,
        [("1", "shall i send you the $20 now?"), ("2", "yes please"), ("1", "sending now")])
    add(S + ": ride re-sent same route, 2nd 'sure' quiet (ruling 2)", [], None,
        [("1", "book me a cab to the airport"), ("2", "sure"), ("1", "book me a cab to the airport"), ("2", "yes please")])
    add(S + ": ride re-sent new route fires (ruling 2)", [], "ride",
        [("1", "book me a cab to the airport"), ("2", "sure"), ("1", "actually book it to the station instead"), ("2", "sure")])
    return cases


def run(url, cases):
    s = requests.Session()
    tag = int(time.time())
    for n, c in enumerate(cases):
        room = f"dm_rs{tag}_{n}" if c["kind"] == "dm" else f"group_rs{tag}_{n}"
        d, earlier = {}, []
        for i, (sender, text) in enumerate(c["msgs"]):
            d = s.post(url, json={"text": text, "room_id": room, "sender": sender,
                                  "message_id": f"{room}_{i}"}, timeout=90).json()
            f = next((x for x in (d.get("intents") or []) if x in ("money", "ride")), None)
            if f and i < len(c["msgs"]) - 1:
                earlier.append(i)
        # A prompt on an earlier message (the offer, or the request alone) is not a hit:
        # FIRING_RULE puts it on the acceptance. Recorded so it is visible, not hidden.
        c["fired_earlier"] = earlier
        fired_all = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
        conv = (d.get("conversation_state") or {}).get("scores") or {}
        c["got"] = fired_all[0] if len(fired_all) == 1 else (fired_all or None)
        c["score"] = conv.get(c["want"] or "money", 0) if c["want"] else max(conv.get("money", 0), conv.get("ride", 0))
        # EXACT: firing ride on top of the right money prompt is wrong (2026-09-17; it used to pass)
        c["ok"] = fired_all == ([c["want"]] if c["want"] else [])
        c["split"] = ("heldout" if any(L.is_heldout(p) for p in c["tested"]) else "train") if c["tested"] else "-"
    return cases


def report(cases, name, show):
    fam = defaultdict(lambda: {"train": [0, 0], "heldout": [0, 0], "-": [0, 0], "early": 0})
    for c in cases:
        f = fam[c["family"]][c["split"]]
        f[0] += c["ok"]; f[1] += 1
        fam[c["family"]]["early"] += bool(c.get("fired_earlier"))
    print(f"\n  robustness suite - {name} - {len(cases)} cases")
    print(f"  {'family':58}{'train phrasings':>18}{'held-out phrasings':>22}{'fired early':>13}")
    tot = {"pos": [0, 0], "neg": [0, 0], "train": [0, 0], "heldout": [0, 0]}
    for k in sorted(fam):
        v = fam[k]
        cell = lambda s: f"{v[s][0]}/{v[s][1]} {100*v[s][0]/v[s][1]:5.1f}%" if v[s][1] else "-"
        if k.startswith("S"):
            print(f"  {k:58}{cell('-'):>40}{v['early']:>13}")
        else:
            print(f"  {k:58}{cell('train'):>18}{cell('heldout'):>22}{v['early']:>13}")
    for c in cases:
        side = "pos" if c["want"] else "neg"
        tot[side][0] += c["ok"]; tot[side][1] += 1
        if c["split"] in ("train", "heldout"):
            tot[c["split"]][0] += c["ok"]; tot[c["split"]][1] += 1
    pct = lambda a: f"{a[0]}/{a[1]} = {100*a[0]/max(a[1],1):.1f}%"
    print(f"\n  must fire, fired:        {pct(tot['pos'])}")
    print(f"  must stay quiet, quiet:  {pct(tot['neg'])}")
    print(f"  train phrasings:         {pct(tot['train'])}")
    print(f"  held-out phrasings:      {pct(tot['heldout'])}")
    if show:
        bad = [c for c in cases if not c["ok"]]
        print(f"\n  first {min(show, len(bad))} of {len(bad)} failures:")
        for c in bad[:show]:
            print(f"    [{c['family'][:30]}|{c['split']}] want={c['want']} got={c['got']} score={c['score']:.3f}  "
                  + "  //  ".join(f"{s}: {t}" for s, t in c["msgs"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8971/detect")
    ap.add_argument("--name", default="model")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    cases = run(a.url, build())
    report(cases, a.name, a.show)
    out = ROOT / f"data/eval/ROBUSTNESS_{a.name}.json"
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
