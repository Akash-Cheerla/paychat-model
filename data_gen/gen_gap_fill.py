"""Generate training data for the three gaps measured on 2026-08-09.

Each gap was found by probing the live model, not guessed:

  1. COMMITMENT WITHOUT AN ACTION VERB. "ill cover it" scores 0.05, "ill cover it,
     sending now" scores 1.00 — same intent, same context, one clause apart. This is
     the single biggest money gap: it explains the failed handoff after a retraction,
     the offer-to-help-after-refusal case (0/2 in behaviour_eval), and "i can cover it"
     sitting at 0.54.

  2. THE "let me <verb> you <amount>" FRAME. 1/8 in behaviour_eval. Fires for send,
     transfer, pay, venmo, upi, phonepe — and not for give, gpay, paypal, cashapp,
     zelle. Three of those four dead apps are the ones the product actually supports.

  3. BIG GROUPS WITH SPLITS. 4-6 speaker rooms are 13% of training. Splits phrased as
     a total need a headcount, which now arrives as the `participants` field, so the
     per-person share is finally learnable rather than regexed.

The KEY LINE of every conversation is supplied from a curated pool here, not written by
DeepSeek — that is what guarantees the target pattern is present and that its label is
right. DeepSeek writes only the surrounding turns. Labels come from the skeleton, the
same approach used for behaviour_eval, because role-derived labels are what made
group_eval_v6 unreliable.

Run:  DEEPSEEK_API_KEY=... python data_gen/gen_gap_fill.py --n 3000
"""
import argparse, json, os, random, re, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.deepseek.com/chat/completions"
KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# ── Gap 1: commitment with no payment verb anywhere in the sentence ──
NO_VERB_COMMIT = [
    "ill cover it", "ill cover this one", "i got this one", "i got this",
    "ill take care of it", "ill handle it", "ill sort it out", "ill sort this",
    "ill do it", "ill do it then", "leave it to me", "consider it done",
    "its on me", "this ones on me", "my treat", "ill get this one",
    "i can cover it", "i can do that", "ill manage it", "ill deal with it",
    "dont worry about it, i got it", "no worries i got it", "ill take it",
    "ill look after it", "count me in for it", "ill front it",
    "ill help you with 500", "let me help you with 20", "let me help with the rent",
    "i can help with 200", "ill chip in for it", "ill put it in",
]
# Same shape, but must NOT fire — no commitment, just sympathy or a maybe.
NO_VERB_NEGATIVE = [
    "i wish i could cover it", "i might be able to cover it",
    "someone should cover it", "can anyone cover it",
    "i covered it last time", "you should have covered it",
    "ill cover it next month", "ill cover it tomorrow",
    "i would cover it but im broke", "maybe i can cover it later",
]

# ── Gap 2: the "let me <verb> you <amount>" frame ──
PAY_VERBS = ["send", "transfer", "pay", "give", "venmo", "paypal", "cashapp",
             "zelle", "gpay", "upi", "paytm", "phonepe"]
LET_ME_FRAMES = ["let me {v} you {amt}", "lemme {v} you {amt}",
                 "let me {v} you the {amt}", "ill {v} you {amt}",
                 "im {v}ing you {amt}", "let me just {v} you {amt}"]
AMOUNTS = ["20", "50", "100", "150", "200", "500", "1000", "2000",
           "20 bucks", "500 rupees", "$40", "₹300", "hundred bucks"]

# ── Gap 4: an OFFER phrased as a question, then accepted ──
# The fire belongs on the ACCEPTANCE, and the payer is the person who offered — the
# reverse of a request, where the accepter pays. Reported live: "can i pay you 100
# rupees" / "Sure" put the payment sheet in front of the person receiving the money.
OFFER_Q = ["shall i send you {amt}", "shall i pay you {amt}", "can i send you {amt}",
           "can i pay you {amt}", "want me to send you {amt}",
           "should i send you {amt} now", "shall i transfer you {amt}",
           "do you want me to send {amt}", "shall i {app} you {amt}",
           "can i {app} you {amt}"]
OFFER_ACCEPT = [
    "sure", "sure thing", "yes please", "yeah please", "please do", "go for it",
    "yeah that would help", "that would help a lot", "yes please send",
    "yeah go ahead", "that would be great thanks", "sure if you dont mind",
    "yes please, thanks a lot", "ok yeah", "yep that works", "if you can, yes",
    "honestly that would save me", "yeah do it", "please, that would be great",
    "k yes", "yes if its no trouble", "sure, appreciate it", "that works, thanks",
    "yeah send it whenever", "ok that helps", "yes go ahead", "sure, thank you",
    "please yes", "yeah if you dont mind", "that would be perfect",
    "yes that would sort me out", "ok cool yes", "yeah id appreciate that",
]
OFFER_DECLINE = [
    "nah im good", "no need really", "dont worry about it", "nah keep it",
    "its fine honestly", "no thats ok", "nah its fine", "dont bother",
    "no its alright", "nah you already got last time", "honestly no need",
    "nah we're square", "no its on me", "dont worry, we're even",
    "nah dont, seriously", "its ok, forget it", "no really its fine",
]

SCENES = ["dinner last night", "the cab fare", "movie tickets", "the groceries",
          "your half of rent", "the concert tickets", "lunch yesterday",
          "the hotel booking", "petrol money", "the birthday gift"]

PROMPT_AROUND = """Write a short, natural chat conversation in English between friends.

SETTING: {setting}
CONTEXT: they are talking about {scene}.

The conversation must end with this EXACT message, unchanged, from {who}:
"{key}"

Write {n} messages BEFORE it that lead up to it naturally. Real texting: lowercase,
contractions, occasional typos, no emoji. English only, no Hindi or Hinglish.
{roles}

Format each line as "SPEAKER: message". Do not number them. Do not add anything
after the final message. Return exactly {total} lines."""


def call(body, temp=1.2):
    try:
        r = requests.post(API, headers={"Authorization": f"Bearer {KEY}"},
                          json={"model": "deepseek-chat",
                                "messages": [{"role": "user", "content": body}],
                                "temperature": temp, "max_tokens": 600},
                          timeout=90)
        if r.status_code != 200:
            return None
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


HIN = ("bhai", "yaar", "paisa", "kitna", "acha", "theek", "haan", "nahi", "kya", "arre")


def parse(txt, speakers):
    out = []
    for l in (txt or "").splitlines():
        l = l.strip()
        m = re.match(r"^\**([A-Z])\**\s*[:\-]\s*(.+)$", l)
        if m and m.group(1) in speakers:
            out.append((m.group(1), m.group(2).strip().strip('"')))
    return out


def build(spec):
    idx, gap = spec
    rnd = random.Random(idx * 104729)

    if gap == "no_verb":
        fires = rnd.random() > 0.28
        key = rnd.choice(NO_VERB_COMMIT if fires else NO_VERB_NEGATIVE)
        group = rnd.random() > 0.45
        speakers = ["A", "B", "C", "D", "E", "F"][:rnd.randint(3, 6)] if group else ["A", "B"]
        who = "B"
        lead = rnd.randint(2, 4)
        roles = ("A is asking for money. B is the one who might pay."
                 + (" The others are just chatting." if group else ""))
        key_role = "ack_money" if fires else "neutral"
        fire = ["money"] if fires else []

    elif gap == "let_me":
        v = rnd.choice(PAY_VERBS)
        key = rnd.choice(LET_ME_FRAMES).format(v=v, amt=rnd.choice(AMOUNTS))
        group = rnd.random() > 0.6
        speakers = ["A", "B", "C", "D"][:rnd.randint(3, 4)] if group else ["A", "B"]
        who = "B"
        lead = rnd.randint(1, 3)
        roles = "B is the one who will pay. A is owed the money."
        key_role, fire = "self_money", ["money"]

    elif gap == "offer_q":
        # A offers, B answers. The fire is on B's ANSWER, and A is the payer.
        accepts = rnd.random() > 0.3
        app = rnd.choice(["paypal", "gpay", "cashapp", "zelle", "venmo", "upi"])
        offer = rnd.choice(OFFER_Q).format(amt=rnd.choice(AMOUNTS), app=app)
        key = rnd.choice(OFFER_ACCEPT if accepts else OFFER_DECLINE)
        speakers = ["A", "B"]
        who = "B"
        lead = 0
        roles = "A is offering to pay B. B answers."
        key_role = "ack_money" if accepts else "reject"
        fire = ["money"] if accepts else []
        # A longer, varied lead-in. With a 2-turn conversation and a fixed answer, the
        # first audit came back at 52% unique text — the model would learn the exact
        # acceptance tokens rather than what an acceptance is.
        body = (f'Write a short chat between 2 friends in English about '
                f'{rnd.choice(SCENES)}.\n'
                f'Start with {rnd.randint(2,4)} messages of natural back-and-forth '
                f'between A and B about it — how it went, who paid, what it cost. '
                f'Vary the wording; do not use stock phrases.\n'
                f'Then A sends exactly: "{offer}"\n'
                f'Then B replies with exactly: "{key}"\n'
                f'Real texting: lowercase, contractions, occasional typos, no emoji. '
                f'English only.\n'
                f'Format "SPEAKER: message", one per line, nothing else.')
        raw = call(body)
        parsed = parse(raw, {"A", "B"})
        if len(parsed) < 2 or parsed[-1][0] != "B":
            return None
        if parsed[-1][1].lower().strip(" .!") != key.lower().strip(" .!"):
            return None
        if any(h in t.lower().split() for _, t in parsed for h in HIN):
            return None
        turns = []
        for i, (sp, tx) in enumerate(parsed):
            last = (i == len(parsed) - 1)
            turns.append({"sender": sp, "text": tx,
                          "role": key_role if last else "offer_money",
                          "fire": list(fire) if last else []})
        return {"scenario": "gap_offer_q", "market": "mixed",
                "relationship": "friends", "participants": 2, "turns": turns}

    else:  # big_split
        n_people = rnd.randint(4, 6)
        speakers = ["A", "B", "C", "D", "E", "F"][:n_people]
        total = rnd.choice([1200, 2400, 3000, 4000, 5000, 6000, 9000])
        per = total // n_people
        style = rnd.random()
        if style < 0.4:
            first = f"{rnd.choice(SCENES)} came to {total} total, send me your shares"
        elif style < 0.7:
            first = f"{rnd.choice(SCENES)} was {total}, thats {per} each"
        else:
            first = f"{rnd.choice(SCENES)} cost {total}, split {n_people} ways"
        # SEVERAL members pay, not one. A split is owed by everybody, and a
        # conversation that only ever shows one payer teaches the opposite: the first
        # audit of this file found 0 conversations where two different people fire,
        # which is precisely the behaviour the gap exists to teach.
        # Some members pay, some do not. A set where every split fires teaches "splits
        # always fire" — the first audit came back 0% negatives and 21% unique text,
        # both of which push the model toward over-firing on any split-shaped message.
        PAYS = [
            "sending mine now", "sending my share", "ok sending mine",
            "sending my part now", "mine's going now", "sending mine too",
            "same, sending mine", "ok mine's on the way", "doing mine now",
            "sending my bit now", "transferring mine now", "mine's done in a sec",
            "ok doing mine", "sending it over now", "putting mine in now",
            "yep sending mine", "on it, sending mine", "sending my portion",
            "ok mine's going", "here's mine, sending", "sending my cut now",
            "alright sending mine", "mine coming now", "sending it now",
            "ok ill send mine now", "sending you mine", "doing my share now",
            "cool, sending mine", "sending mine right now", "mine's on its way",
            "paying mine now", "ok transferring my share", "sending my half now",
        ]
        # Look like a payment, commit to nothing. These must NOT fire.
        DEFERS = [
            "ill do mine tonight", "mine after payday", "can i send tomorrow",
            "ill sort mine later", "give me till friday", "mine's coming later",
            "ill do it when i get home", "next week for mine ok?",
            "i paid mine last time though", "wait how much is mine again",
            "hold on, mine's already done from before",
        ]
        rnd.shuffle(PAYS); rnd.shuffle(DEFERS)
        nobody_pays = rnd.random() < 0.15
        if nobody_pays:
            n_payers = 0
            n_defer = rnd.randint(2, min(3, n_people - 1))
        else:
            n_payers = rnd.randint(2, min(4, n_people - 1))
            n_defer = 1 if (rnd.random() < 0.4 and n_people - 1 > n_payers) else 0
        members = speakers[1:]
        rnd.shuffle(members)
        key_lines = [(sp, PAYS[i], True) for i, sp in enumerate(members[:n_payers])]
        key_lines += [(sp, DEFERS[i], False)
                      for i, sp in enumerate(members[n_payers:n_payers + n_defer])]
        rnd.shuffle(key_lines)
        if not key_lines:
            return None
        who = None
        lead = 0
        roles = f"A asks the group of {n_people} to pay their shares."
        key_role, fire = "ack_money", ["money"]

    setting = ("a group chat with %d friends" % len(speakers)) if len(speakers) > 2 \
              else "a one-to-one chat between 2 friends"
    if gap == "big_split":
        seq = "\n".join(f'{sp}: {tx}' for sp, tx, _ in key_lines)
        body = (f'Write a group chat between {len(speakers)} friends '
                f'({", ".join(speakers)}) in English.\n'
                f'The FIRST message, from A, must be exactly: "{first}"\n'
                f'Then 1 or 2 short reactions from people NOT in the list below '
                f'(a joke or a question) — neither may commit to paying.\n'
                f'Then these messages, in this order, EXACTLY as written:\n{seq}\n'
                f'You may add one short reaction between them, from someone not in '
                f'that list, that does not commit to paying.\n'
                f'Real texting: lowercase, contractions, no emoji. English only.\n'
                f'Format "SPEAKER: message", one per line, nothing else.')
        raw = call(body)
        parsed = parse(raw, set(speakers))
        if len(parsed) < 1 + len(key_lines):
            return None
        want = [(sp, tx.lower().strip(" .!")) for sp, tx, _ in key_lines]
        got = [(sp, tx.lower().strip(" .!")) for sp, tx in parsed]
        # every payer line must be present, in order, unaltered
        pos, order_ok = -1, True
        for w in want:
            try:
                pos = got.index(w, pos + 1)
            except ValueError:
                order_ok = False
                break
        if not order_ok:
            return None
        if any(h in t.lower().split() for _, t in parsed for h in HIN):
            return None
        paying = {(sp, tx.lower().strip(" .!")) for sp, tx, f in key_lines if f}
        turns = [{"sender": sp, "text": tx,
                  "role": "ack_money" if (sp, tx.lower().strip(" .!")) in paying else
                          ("request_money" if i == 0 else "neutral"),
                  "fire": ["money"] if (sp, tx.lower().strip(" .!")) in paying else []}
                 for i, (sp, tx) in enumerate(parsed)]
        return {"scenario": "gap_big_split", "market": "mixed",
                "relationship": "friends", "participants": len(speakers),
                "turns": turns}
    else:
        body = PROMPT_AROUND.format(setting=setting, scene=rnd.choice(SCENES),
                                    who=who, key=key, n=lead, roles=roles,
                                    total=lead + 1)
    raw = call(body)
    parsed = parse(raw, set(speakers))
    if len(parsed) < 2:
        return None
    if parsed[-1][0] != who or parsed[-1][1].lower().strip(" .!") != key.lower().strip(" .!"):
        return None
    if any(h in t.lower().split() for _, t in parsed for h in HIN):
        return None
    # the lead-up must not itself commit, or the label on the key line is not the
    # only thing firing in the conversation
    LEAD_COMMIT = re.compile(r"\b(sending|paying|transferring|venmoing)\s+(it|now|you)", re.I)
    if any(LEAD_COMMIT.search(t) for _, t in parsed[:-1]):
        return None

    turns = []
    for i, (sp, tx) in enumerate(parsed):
        last = (i == len(parsed) - 1)
        turns.append({"sender": sp, "text": tx,
                      "role": key_role if last else ("request_money" if i == 0 else "neutral"),
                      "fire": list(fire) if last else []})
    return {"scenario": f"gap_{gap}", "market": "mixed", "relationship": "friends",
            "participants": len(speakers), "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="target conversations")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="data/conversations/gap_fill_v6.json")
    a = ap.parse_args()
    if not KEY:
        sys.exit("DEEPSEEK_API_KEY not set")

    mix = os.environ.get("GAPMIX","no_verb:5,let_me:3,big_split:2")
    mix = [k for part in mix.split(",") for k, n in [part.split(":")] for _ in range(int(n))]
    jobs = [(i, mix[i % len(mix)]) for i in range(int(a.n * 4))]
    rows, seen = [], set()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(build, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                sig = "|".join(t["text"] for t in r["turns"])
                if sig not in seen:
                    seen.add(sig); rows.append(r)
            if k % 300 == 0:
                print(f"  {k} requested, {len(rows)} kept", flush=True)
            if len(rows) >= a.n:
                break

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    turns = sum(len(r["turns"]) for r in rows)
    fires = sum(1 for r in rows for t in r["turns"] if t["fire"])
    print(f"\n  wrote {len(rows)} conversations ({turns} turns, {fires} firing) -> {out}")
    print(f"    by gap     : {dict(Counter(r['scenario'] for r in rows))}")
    print(f"    speakers   : {dict(sorted(Counter(r['participants'] for r in rows).items()))}")


if __name__ == "__main__":
    main()
