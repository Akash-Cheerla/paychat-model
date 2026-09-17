"""Build conv_windows_v17b.zip: v17 with its regressions fixed.

v17 (2026-09-17) beat v14 on both real logs but lost behaviours the batteries cover
(data/eval/V17_RESULTS.md). Causes were in v17's batch, so v17b changes only data:

  base   conv_windows_v16s.zip = v16r + data_gen/rulings_0917_relabel.py (debt + "sending now"
         labelled quiet in 41/10/8 windows, FIRING_RULE s1a and ruling 8)
  A8     second group answerer: the FIRST answer is now a bare sure/ok. v17 drew it from the
         action acks too ("on it, booking now"), which taught that a third person's bare ok
         after a clear take fires - the g09 false prompt, against ruling 6/7 (2026-09-17)
  C1-C11 new families, below

  C1/C2  self-initiated send / book, no request                     fire  (s3; v17 lost "i'll send it")
  C3     future promise alone, no request                           quiet (s3, the contrast to C1)
  C4/C5  debt stated -> agrees to pay / only admits it              fire / quiet  (ruling 8)
  C6/C7  shortfall -> polite noise / a commitment                   quiet / fire  (s1a)
  C8     clear take ("let me book a cab") answering a request       fire  (s3c; v17: 0.996 -> 0.218)
  C9     clear take, then a third person's bare ok                  quiet (ruling 6)
  C10    clear take, then a third person's clear words              fire  (ruling 6)
  C11    two open requests: the verb decides, else the most recent  fire, that intent only (s6a)

v17's original docstring follows.

Build conv_windows_v17.zip: v16's windows + the 2026-09-15 relabel + coverage windows.

Why. v14's real-traffic misses were not random. Probes showed the classifier flips on one
word ("10₹" vs "$10", "oh ok"/"yep fr" vs "ok", "shall i send...?" offers), and its
training data had zero fire examples of those phrasings. More of the same generators
would teach more exact phrasings. This batch varies ONE factor per family across many
phrasings, from tests/robustness_lexicon.py's TRAIN half only - the HELDOUT half, which
includes every phrasing seen in dogfood, never appears here, so the robustness suite
measures whether the model learned the rule rather than the words.

Families (fire on the last message unless marked quiet):
  A1/A2  money / ride request -> acceptance wording
  A3/A4  request wording and amount formats -> plain acceptance
  A5     agreement + detail question            (ruling 1, 2026-09-15)
  A6     offer -> acceptance                     (FIRING_RULE s3c: fires on the acceptance)
  A7     interruption between request and acceptance
  A8     second group answerer                   (ruling 3)
  A9     same request re-sent after a failure    (ruling 5)
  B1-B6  decline / question only / future / already done / hedge / looks-like-yes  (quiet)
  B7     requester's own follow-up                                                  (quiet)
  B8     acceptance words with no request                                           (quiet)
  B9     offer alone, offer declined                                                (quiet)
  B10    same request re-sent, accepted again    (ruling 2)                         (quiet)
  B11    answer after someone confirmed it done  (ruling 3)                         (quiet)
  B12    request alone                                                              (quiet)

Filler chat comes from v16's own neutral windows so new windows look like old ones.
A 5% slice goes to val (threshold selection); test only gets the relabel.

    python data_gen/build_conv_v17.py            # writes conv_windows_v17.zip
"""
import hashlib, io, json, random, re, sys, zipfile
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import robustness_lexicon as L   # noqa: E402

SRC = ROOT / "conv_windows_v16s.zip"      # v16 + rulings_0915_relabel.py + rulings_0917_relabel.py
OUT = ROOT / "conv_windows_v17b.zip"
R = random.Random(17)
VAL_SHARE = 0.05

T = {k: L.split(v)[0] for k, v in L.LISTS.items()}     # TRAIN half of every list
HELD = set(p for v in L.LISTS.values() for p in L.split(v)[1])
FAILURE = ["did not get an intent", "it didn't go through", "not working on my side", "payment failed",
           "didn't work, try again", "the app didn't show anything", "hmm nothing came up"]
MONEY_WORDS = re.compile(r"\b(send|pay|lend|spot|venmo|paytm|gpay|upi|zelle|transfer|wire|borrow|cash|bucks|dollars|rupees|rs|inr|money)\b|[$₹\d]", re.I)
# Filler must carry no payment/booking signal of its own: no completions, owing, trips,
# cancellations or short acknowledgements that could read as an answer.
FILLER_BAN = re.compile(r"\b(paid|pay\w*|send\w*|sent|done|owe\w*|broke|split\w*|bill\w*|price|cost\w*|fare|trip|driver|"
                        r"pick\w*|drop\w*|airport|station|cancel\w*|book\w*|order\w*|deal|ok|okay|sure|yes|yeah|yep|k|kk|"
                        r"bet|got|on it|tomorrow|friday|later|promise|help|need\w*|please|pls|rent|landlord|loan|"
                        r"expensive|cheap|rush|cant|can't|lied|thank\w*|thx|ty|transfer\w*|wallet|card|ticket\w*|"
                        r"ride|lift|taxi|uber|ola|cab|money|cash|bucks|rs|asking|ask\w*)\b", re.I)
RIDE_WORDS = re.compile(r"\b(cab|uber|ola|lyft|taxi|ride|rapido|auto|book)\b", re.I)


def _held_re(h):
    pat = re.escape(h.lower())
    for k, v in ((r"\{amt\}", r"\S+(?: \S+)?"), (r"\{place\}", r".+"), (r"\{a\}", r".+"), (r"\{b\}", r".+"), (r"\{n\}", r"[\d,]+")):
        pat = pat.replace(k, v)
    # a template with an amount only matches text that has a number ("can you give me the
    # total?" is not "can you give me {amt}")
    return re.compile("^" + pat + "$"), ("{amt}" in h or "{n}" in h)


HELD_PATS = [_held_re(h) for h in HELD]


def is_held_text(t):
    s = t.lower().strip(" .!👍🙏")
    return any(p.match(s) and (not needs_digit or re.search(r"\d", s)) for p, needs_digit in HELD_PATS)


def fillers(train):
    pool = set()
    for w in train:
        if w.get("role") != "neutral" or w["fire"]:
            continue
        for m in w["window"]:
            t = m["t"].strip()
            if (3 <= len(t.split()) <= 12 and not MONEY_WORDS.search(t) and not RIDE_WORDS.search(t)
                    and not FILLER_BAN.search(t) and not is_held_text(t)):   # v17b: e.g. "i haven't forgotten"
                pool.add(t)
    return sorted(pool)


def vary(t):
    r = R.random()
    if r < 0.35:
        return t
    if r < 0.55:
        return t[:1].upper() + t[1:]
    if r < 0.70:
        return t + R.choice([".", "!", " 👍", " 🙏", "!!"])
    if r < 0.85:
        return t.lower()
    return t.upper() if len(t) < 12 else t


def amt():
    return R.choice(T["AMOUNTS"]).format(n=R.choice(L.NUMBERS))


def money_req(tpl=None):
    return (tpl or R.choice(T["MONEY_REQUESTS"])).format(amt=amt())


def ride_req(tpl=None):
    a, b = R.sample(L.PLACES, 2)
    return (tpl or R.choice(T["RIDE_REQUESTS"])).format(place=R.choice(L.PLACES), a=a, b=b)


def main():
    z = zipfile.ZipFile(SRC)
    splits = {s: json.loads(z.read(s)) for s in ("train.json", "val.json", "test.json")}
    pool = fillers(splits["train.json"])
    print(f"  base {SRC.name}: train {len(splits['train.json']):,}  filler pool {len(pool):,}")
    new = []

    def pre(k_max=4, speakers=("1", "2")):
        return [{"s": R.choice(speakers), "t": R.choice(pool)} for _ in range(R.randint(0, k_max))]

    def add(family, role, fire, msgs, group=False):
        spk = ("1", "2", "3") if group else ("1", "2")
        win = pre(speakers=spk) + [{"s": s, "t": vary(t)} for s, t in msgs]
        win = win[-10:]
        text = " ".join(m["t"] for m in win)
        market = "India" if re.search(r"₹|\brs\b|rupees|\binr\b|paytm|gpay|upi|ola|rapido", text, re.I) else "US"
        new.append({"window": win, "fire": fire, "role": role, "conv": hashlib.sha1(text.encode()).hexdigest()[:12],
                    "market": market, "scenario": f"v17_{family}"})

    BARE_YES = [p for p in T["ACCEPT"] if p in ("ok", "okay", "sure", "yes", "yeah", "ya", "yep", "yup", "k", "kk")]
    money_acks = T["ACCEPT"] + T["ACCEPT_MONEY_ACTION"]
    ride_acks = T["ACCEPT"] + T["ACCEPT_RIDE_ACTION"]
    for p in money_acks:
        for _ in range(60):
            add("A1_money_ack", "ack_money", ["money"], [("1", money_req()), ("2", p)])
    for p in ride_acks:
        for _ in range(50):
            add("A2_ride_ack", "ack_ride", ["ride"], [("1", ride_req()), ("2", p)])
    for tpl in T["MONEY_REQUESTS"]:
        for _ in range(80):
            add("A3_money_request", "ack_money", ["money"], [("1", money_req(tpl)), ("2", R.choice(T["ACCEPT"]))])
    for tpl in T["RIDE_REQUESTS"]:
        for _ in range(100):
            add("A4_ride_request", "ack_ride", ["ride"], [("1", ride_req(tpl)), ("2", R.choice(T["ACCEPT"]))])
    # Ruling 1 has to overturn a prior the model learned from ~3,000 base windows where an
    # agreement plus a question stays quiet, so this family gets weight, not a token amount.
    for p in T["ACCEPT_WITH_QUESTION_MONEY"]:
        for _ in range(300):
            add("A5_agree_question", "question", ["money"], [("1", money_req()), ("2", p)])
    for p in T["ACCEPT_WITH_QUESTION_RIDE"]:
        for _ in range(200):
            add("A5_agree_question", "question", ["ride"], [("1", ride_req()), ("2", p)])
    for o in T["OFFERS_MONEY"]:
        for _ in range(80):
            add("A6_offer_accept", "ack_after_offer", ["money"], [("1", o.format(amt=amt())), ("2", R.choice(T["OFFER_ACCEPT"]))])
    for o in T["OFFERS_RIDE"]:
        for _ in range(80):
            add("A6_offer_accept", "ack_after_offer", ["ride"], [("1", o), ("2", R.choice(T["OFFER_ACCEPT"]))])
    for x in T["INTERRUPT_BY_ACCEPTOR"]:
        for _ in range(80):
            if R.random() < 0.5:
                add("A7_interrupt", "ack_money", ["money"], [("1", money_req()), ("2", x), ("2", R.choice(money_acks))])
            else:
                add("A7_interrupt", "ack_ride", ["ride"], [("1", ride_req()), ("2", x), ("2", R.choice(ride_acks))])
    for x in T["INTERRUPT_BY_REQUESTER"]:
        for _ in range(80):
            add("A7_interrupt", "ack_money", ["money"], [("1", money_req()), ("1", x), ("2", R.choice(money_acks))])
    # Ruling 7: B's answer is a BARE yes, so C's yes is also answering the request. After a clear
    # take by B, C's bare ok is acknowledgement instead - that is C9, quiet.
    for _ in range(600):
        if R.random() < 0.5:
            add("A8_second_answerer", "ack_money", ["money"],
                [("1", money_req().replace("can you", "can someone", 1)), ("2", R.choice(BARE_YES)), ("3", R.choice(money_acks))], group=True)
        else:
            add("A8_second_answerer", "ack_ride", ["ride"],
                [("1", ride_req()), ("2", R.choice(BARE_YES)), ("3", R.choice(ride_acks))], group=True)
    for _ in range(300):
        q = money_req() if R.random() < 0.5 else ride_req()
        intent = "money" if MONEY_WORDS.search(q) and not re.search(r"\bbook|\bcab|uber|ola|lyft|taxi|rapido", q, re.I) else "ride"
        ack = R.choice(money_acks if intent == "money" else ride_acks)
        add("A9_retry_after_failure", f"ack_{intent}", [intent], [("1", q), ("2", R.choice(T["ACCEPT"])), (R.choice("12"), R.choice(FAILURE)), ("1", q), ("2", ack)])

    def quiet_after_request(family, role, phrases, n, msg_sender="2"):
        for p in phrases:
            for _ in range(n):
                q = money_req() if ("book" not in p and R.random() < 0.6) else ride_req()
                add(family, role, [], [("1", q), (msg_sender, p)])

    quiet_after_request("B1_decline", "decline", T["DECLINE"], 100)
    quiet_after_request("B2_question_only", "question", T["QUESTION_ONLY_MONEY"] + T["QUESTION_ONLY_RIDE"], 100)
    quiet_after_request("B3_future", "future_promise", T["FUTURE"], 100)
    quiet_after_request("B4_already_done", "already_done", T["DONE"], 100)
    quiet_after_request("B5_hedge", "hedge", T["HEDGE"], 100)
    quiet_after_request("B6_hard_negative", "decline", T["HARD_NEGATIVE"], 100)
    quiet_after_request("B7_requester_followup", "chase", T["REQUESTER_FOLLOWUP"], 100, msg_sender="1")
    for p in T["ACCEPT"]:
        for _ in range(20):
            add("B8_ack_no_request", "neutral", [], [("1", R.choice(T["NON_REQUEST_OPENERS"] + pool)), ("2", p)])
    for o in T["OFFERS_MONEY"] + T["OFFERS_RIDE"]:
        for _ in range(60):
            add("B9_offer_alone", "offer", [], [("1", o.format(amt=amt()))])
        for _ in range(40):
            add("B9_offer_declined", "decline", [], [("1", o.format(amt=amt())), ("2", R.choice(T["DECLINE"] + T["HARD_NEGATIVE"]))])
    for _ in range(400):
        if R.random() < 0.5:
            q, acks = money_req(), money_acks
        else:
            q, acks = ride_req(), ride_acks
        add("B10_resent_same_request", "ack_other", [], [("1", q), ("2", R.choice(acks)), ("1", q), ("2", R.choice(T["ACCEPT"]))])
    for _ in range(300):
        if R.random() < 0.5:
            add("B11_after_done", "ack_other", [],
                [("1", money_req().replace("can you", "can someone", 1)), ("2", R.choice(money_acks)), ("2", R.choice(["sent it", "done, sent", "paid"])), ("3", R.choice(T["ACCEPT"]))], group=True)
        else:
            add("B11_after_done", "ack_other", [],
                [("1", ride_req()), ("2", R.choice(ride_acks)), ("2", R.choice(["booked it", "done, booked", "booked, check your app"])), ("3", R.choice(T["ACCEPT"]))], group=True)
    for _ in range(500):
        add("B12_request_alone", "request_money" if R.random() < 0.5 else "request_ride", [],
            [("1", money_req() if R.random() < 0.5 else ride_req())])

    # ---------------- v17b families (2026-09-17) ----------------
    group_money = lambda: "can someone " + R.choice(["send me {amt}", "venmo me {amt}", "spot me {amt}", "cover {amt} for me",
                                                      "gpay me {amt}", "lend me {amt} till friday"]).format(amt=amt())
    group_ride = lambda: "can someone " + R.choice(["book me a cab to {p}", "get me an uber to {p}", "order me an ola to {p}",
                                                     "book a taxi for me to {p}"]).format(p=R.choice(L.PLACES))
    fill = lambda k=3: [{"s": R.choice(("1", "2")), "t": R.choice(pool)} for _ in range(R.randint(0, k))]

    def add_pre(family, role, fire, pre_msgs, msgs, group=False):
        """Like add(), but with fixed messages before the tested ones (no random filler in between)."""
        win = pre_msgs + [{"s": s, "t": vary(t)} for s, t in msgs]
        win = win[-10:]
        text = " ".join(m["t"] for m in win)
        market = "India" if re.search(r"₹|\brs\b|rupees|\binr\b|paytm|gpay|upi|ola|rapido", text, re.I) else "US"
        new.append({"window": win, "fire": fire, "role": role, "conv": hashlib.sha1(text.encode()).hexdigest()[:12],
                    "market": market, "scenario": f"v17b_{family}"})

    for p in T["SELF_SEND"]:
        for _ in range(90):
            add_pre("C1_self_send", "self_money", ["money"], fill(), [("1", p.format(amt=amt()))])
    for p in ["sending now", "transferring now"]:                       # TRAIN phrases of ACCEPT_MONEY_ACTION
        for _ in range(60):
            add_pre("C1_self_send", "self_money", ["money"], fill(), [("1", p)])
    for p in T["SELF_BOOK"]:
        for _ in range(100):
            add_pre("C2_self_book", "self_ride", ["ride"], fill(), [("1", p.format(place=R.choice(L.PLACES)))])
    for p in ["booking now", "booking it rn"]:                          # TRAIN phrases of ACCEPT_RIDE_ACTION
        for _ in range(60):
            add_pre("C2_self_book", "self_ride", ["ride"], fill(), [("1", p)])
    for p in T["FUTURE"]:
        for _ in range(100):
            add_pre("C3_future_alone", "future_promise", [], fill(), [("1", p)])
    debt_yes = T["DEBT_PAY_AGREE"] + ["sure", "sure thing", "sure np"]
    debt_admit = T["DEBT_ADMIT"] + ["alright"]
    for d in T["DEBT_STATEMENTS"]:
        for _ in range(110):
            add_pre("C4_debt_pay", "ack_money", ["money"], fill(), [("1", d.format(amt=amt())), ("2", R.choice(debt_yes))])
        for _ in range(90):
            add_pre("C5_debt_admit", "ack_other", [], fill(), [("1", d.format(amt=amt())), ("2", R.choice(debt_admit))])
    # Weighted up after the first CPU probe: "im short 300" / "sure" stayed a false prompt in every
    # arm. Half the replies are the bare yes words that actually fool it.
    short_bare = ["sure", "ok", "okay", "yeah"]
    for s in T["SHORTFALL"]:
        for _ in range(250):
            reply = R.choice(short_bare) if R.random() < 0.5 else R.choice(T["SHORTFALL_POLITE"])
            add_pre("C6_shortfall_polite", "neutral", [], fill(), [("1", s.format(amt=amt())), ("2", reply)])
        for _ in range(60):
            add_pre("C7_shortfall_commit", "ack_money", ["money"], fill(),
                    [("1", s.format(amt=amt())), ("2", R.choice(T["SHORTFALL_COMMIT"]).format(amt=amt()))])
    third_bare = ["okay", "ok", "sure", "cool", "k", "alright", "nice", "great", "yes", "yeah"]
    for intent, takes, req in (("money", T["CLEAR_TAKE_MONEY"], group_money), ("ride", T["CLEAR_TAKE_RIDE"], group_ride)):
        for t in takes:
            for _ in range(60):
                add_pre("C8_clear_take_answers", f"ack_{intent}", [intent], fill(2), [("1", req()), ("2", t)], group=True)
            # weighted up after the first CPU probe: 400 windows did not move g09 against the ~720
            # offer -> accept windows that share its shape once speakers are normalised
            for _ in range(175):
                add_pre("C9_third_bare_after_take", "ack_other", [], fill(2),
                        [("1", req()), ("2", t), ("3", R.choice(third_bare))], group=True)
    for p in T["THIRD_EXPLICIT"]:
        ride = "book" in p
        for _ in range(60):
            add_pre("C10_third_explicit_after_take", "ack_ride" if ride else "ack_money", ["ride" if ride else "money"], fill(2),
                    [("1", group_ride() if ride else group_money()), ("2", R.choice(T["CLEAR_TAKE_RIDE"] if ride else T["CLEAR_TAKE_MONEY"])),
                     ("3", p)], group=True)
    for _ in range(600):
        money_first = R.random() < 0.5
        reqs = ([("1", money_req()), ("3", "also " + ride_req())] if money_first
                else [("3", ride_req()), ("1", "also " + money_req())])
        r = R.random()
        if r < 0.4:
            add_pre("C11_two_open_verb", "ack_money", ["money"], [], reqs + [("2", R.choice(T["ACCEPT_MONEY_ACTION"]))], group=True)
        elif r < 0.8:
            add_pre("C11_two_open_verb", "ack_ride", ["ride"], [], reqs + [("2", R.choice(T["ACCEPT_RIDE_ACTION"]))], group=True)
        else:                                                              # no verb: the most recent request
            add_pre("C11_two_open_recent", "ack_other", ["ride" if money_first else "money"], [],
                    reqs + [("2", R.choice(["ok", "sure", "okay", "yes"]))], group=True)

    # Leakage guard: no held-out phrasing, templates included, may be ANY message of a new window.
    held_hits = [(w["scenario"], m["t"]) for w in new for m in w["window"] if is_held_text(m["t"])]
    assert not held_hits, f"{len(held_hits)} new-window messages match a HELDOUT phrasing, e.g. {held_hits[:3]}"

    R.shuffle(new)
    # Split by CONVERSATION, not by window. `conv` is a hash of the text, so a short window the
    # generator produced twice ("i can book you a cab" / "sure") shares one id; a per-window
    # cut put 12 of those on both sides and the notebook's integrity gate refused to train.
    n_val = int(len(new) * VAL_SHARE)
    size = Counter(w["conv"] for w in new)
    val_convs, taken = set(), 0
    for w in new:
        if taken >= n_val:
            break
        if w["conv"] not in val_convs:
            val_convs.add(w["conv"])
            taken += size[w["conv"]]
    splits["val.json"] += [w for w in new if w["conv"] in val_convs]
    splits["train.json"] += [w for w in new if w["conv"] not in val_convs]
    n_val = taken
    # The same gate the notebook runs (cell "Integrity re-check"), so it fails here, not on Colab.
    ids = {s: {r["conv"] for r in rows} for s, rows in splits.items()}
    for a, b in (("train.json", "test.json"), ("train.json", "val.json"), ("val.json", "test.json")):
        assert not ids[a] & ids[b], f"{a}/{b} conversation overlap: {len(ids[a] & ids[b])}"
    assert not any(not r["window"] for rows in splits.values() for r in rows), "empty window"
    fam = Counter(w["scenario"] for w in new)
    fire = sum(1 for w in new if w["fire"])
    print(f"  new windows {len(new):,}: fire {fire:,}  quiet {len(new) - fire:,}   -> train {len(new) - n_val:,}, val {n_val:,}")
    for k, v in sorted(fam.items()):
        print(f"    {k:28} {v:6,}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zo:
        for s, rows in splits.items():
            zo.writestr(s, json.dumps(rows, ensure_ascii=False))
    tr = splits["train.json"]
    print(f"  wrote {OUT.name}: train {len(tr):,} (fire {100*sum(1 for w in tr if w['fire'])/len(tr):.1f}%), "
          f"val {len(splits['val.json']):,}, test {len(splits['test.json']):,}")
    for w in R.sample(new, 12):
        print(f"    {w['scenario']:26} {str(w['fire']):10} " + "  |  ".join(f"{m['s']}: {m['t'][:40]}" for m in w["window"][-4:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
