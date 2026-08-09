"""Generate a behaviour eval covering every money/ride case we have argued about.

Different from gen_conversations.py in one important way: the LABELS COME FROM THE
SKELETON, not from the model. Each scenario declares its turns, which speaker sends
each one, and which turn fires what. DeepSeek only writes the wording. A model that
misunderstands a scenario produces unnatural text, which we can see and drop — it
cannot produce a silently wrong label, which is the failure mode that made
group_eval_v6's numbers soft.

Covers the cases real users hit and the ones the generated evals never had:
offers ("shall I send you 100"), splits ("send me your shares"), rejection then
revival, handoffs, chasing, and destination changes by the rider vs by the booker.

Run:  DEEPSEEK_API_KEY=... python data_gen/gen_behaviour_eval.py --per 6
"""
import argparse, json, os, random, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.deepseek.com/chat/completions"
KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Each turn: (speaker, what to write, fires)
#   speaker : "A" asks/rides, "B" pays/books, "C" bystander
#   fires   : [] | ["money"] | ["ride"]
# The LAST turn that fires is the one the prompt should appear on.
SCENARIOS = [
 # ── money, simple ───────────────────────────────────────────────
 ("m_request_ack", "easy", "DM", [
   ("A", "asks B to send a specific amount for a specific thing", []),
   ("B", "agrees to send it now", ["money"])]),
 ("m_self_initiated", "easy", "DM", [
   ("A", "mentions they owe B for something", []),
   ("A", "says they are sending the money right now", ["money"])]),
 ("m_rejection", "easy", "DM", [
   ("A", "asks B for money", []),
   ("B", "refuses, they are short of cash", [])]),
 ("m_already_done", "easy", "DM", [
   ("A", "asks whether B sent the money yet", []),
   ("B", "says they already sent it earlier", [])]),
 ("m_future_promise", "easy", "DM", [
   ("A", "asks B for money", []),
   ("B", "promises to send it on a named day next week, not now", [])]),
 ("m_statement_not_request", "easy", "DM", [
   ("A", "states that B still owes them money, without asking for it", []),
   ("B", "reacts casually, agreeing it is true but committing to nothing", [])]),

 # ── money, medium ───────────────────────────────────────────────
 ("m_counter_offer", "medium", "DM", [
   ("A", "asks B for a large amount", []),
   ("B", "counters with a smaller amount and asks if that is ok", []),
   ("A", "accepts the smaller amount", []),
   ("B", "says they are sending it now", ["money"])]),
 ("m_offer_accepted", "medium", "DM", [
   ("A", "offers to pay B a specific amount, phrased as a question", []),
   ("B", "accepts the offer", ["money"])]),
 ("m_offer_let_me", "medium", "DM", [
   ("A", "says let me send you the amount, as a statement not a question", ["money"])]),
 ("m_deferral_then_commit", "medium", "DM", [
   ("A", "asks B for money", []),
   ("B", "says not right now", []),
   ("A", "follows up later asking again", []),
   ("B", "gives in and says they are sending it now", ["money"])]),
 ("m_drift_then_ack", "medium", "DM", [
   ("A", "asks B for a specific amount", []),
   ("B", "changes the subject to something unrelated", []),
   ("A", "replies about that unrelated thing", []),
   ("B", "comes back to it and says they are sending it now", ["money"])]),
 ("m_chasing", "medium", "DM", [
   ("A", "asks B for money", []),
   ("B", "agrees and says sending now", ["money"]),
   ("A", "chases, asking if it went through", []),
   ("B", "confirms it is done", [])]),

 # ── money, hard ─────────────────────────────────────────────────
 ("m_reject_then_revive", "hard", "DM", [
   ("A", "asks B for money", []),
   ("B", "refuses because they are broke", []),
   ("A", "says no worries", []),
   ("B", "changes their mind and says they are sending it now", ["money"])]),
 ("m_reject_then_smaller_offer", "hard", "DM", [
   ("A", "asks B for a specific amount", []),
   ("B", "refuses, they cannot manage that much", []),
   ("B", "offers to help with a smaller amount instead", []),
   ("A", "accepts the smaller amount gratefully", ["money"])]),
 ("m_split_shares", "hard", "GROUP", [
   ("A", "says the trip cost a total amount and asks the group to send their shares", []),
   ("B", "says they are sending their share now", ["money"]),
   ("C", "says they are sending theirs too", ["money"])]),
 ("m_split_covers_two", "hard", "GROUP", [
   ("A", "says dinner cost a total, split between three people", []),
   ("B", "says they will cover their own share and C's as well", ["money"])]),
 ("m_handoff", "hard", "GROUP", [
   ("A", "asks the group for money for something", []),
   ("B", "says they will do it", ["money"]),
   ("B", "backs out, their card is blocked", []),
   ("C", "says they will cover it instead", ["money"])]),
 ("m_two_requests_one_ack", "hard", "GROUP", [
   ("A", "asks the group for money for one thing", []),
   ("C", "asks the group for money for a different thing", []),
   ("B", "agrees, naming which of the two they mean", ["money"])]),
 ("m_group_one_taker", "medium", "GROUP", [
   ("A", "asks if anyone can spot them a small amount", []),
   ("C", "makes a joke, commits to nothing", []),
   ("B", "says they have got it and are sending now", ["money"])]),

 # ── ride, simple ────────────────────────────────────────────────
 ("r_request_ack", "easy", "DM", [
   ("A", "asks B to book them a cab to a named place", []),
   ("B", "agrees and is doing it now", ["ride"])]),
 ("r_self_initiated", "easy", "DM", [
   ("A", "says they are booking a cab right now", ["ride"])]),
 ("r_rejection", "easy", "DM", [
   ("A", "asks B to book them a cab", []),
   ("B", "says no, tells them to walk, it is close", [])]),
 ("r_own_car", "easy", "DM", [
   ("A", "asks B to pick them up in B's own car", []),
   ("B", "agrees to come and get them", [])]),
 ("r_price_talk", "easy", "DM", [
   ("A", "complains that cab prices are high right now", []),
   ("B", "agrees prices are bad", [])]),
 ("r_already_booked", "easy", "DM", [
   ("A", "asks B to book a cab", []),
   ("B", "says they already booked one, it is arriving", [])]),

 # ── ride, medium / hard ─────────────────────────────────────────
 ("r_rider_changes_dest", "medium", "DM", [
   ("A", "asks B to book a cab to a named place", []),
   ("A", "changes their mind to a different named place", []),
   ("B", "agrees and is booking now", ["ride"])]),
 ("r_booker_suggests_not_agreed", "hard", "DM", [
   ("A", "asks B to book a cab to a named place, and says where they are now", []),
   ("B", "suggests a completely different destination instead", []),
   ("B", "says they are booking now, without A ever agreeing", ["ride"])]),
 ("r_where_to", "medium", "DM", [
   ("A", "asks B to book a cab, without saying where", []),
   ("B", "asks where to", []),
   ("A", "names the place", []),
   ("B", "says booking now", ["ride"])]),
 ("r_group_taker", "medium", "GROUP", [
   ("A", "asks if someone can book them a cab, names the place", []),
   ("C", "says they are busy", []),
   ("B", "says they are booking it now", ["ride"])]),
 ("r_discussed_not_booked", "hard", "DM", [
   ("A", "wonders aloud whether to take a cab or the metro", []),
   ("B", "gives an opinion about which is better", []),
   ("A", "says they will decide later", [])]),
]

PROMPT = """Write a short, natural chat conversation in English between friends.

SETTING: {setting}
SCENARIO: {scenario}

Write EXACTLY {n} messages, in this exact order, one per line:
{spec}

Rules:
- Real texting: lowercase is fine, contractions, occasional typos, no emoji.
- English only. No Hindi or Hinglish words.
- Speaker A is the one who wants something. B is the one who would pay or book.{cnote}
- Amounts: use plain numbers with a currency ($ or rupees). Ride places: real city areas.
- Do NOT number the lines. Do NOT add any message beyond the {n}.
- Each line must be ONLY the message text, prefixed by the speaker letter and a colon.

Return exactly {n} lines, nothing else."""


def build(args):
    idx, (name, tier, kind, turns) = args
    rnd = random.Random(idx * 7919)
    spec = "\n".join(f"{sp}: {desc}" for sp, desc, _ in turns)
    cnote = " C is a third person in the group." if kind == "GROUP" else ""
    setting = ("a group chat with 3 friends" if kind == "GROUP"
               else "a one-to-one chat between 2 friends")
    body = PROMPT.format(setting=setting, scenario=name.replace("_", " "),
                         n=len(turns), spec=spec, cnote=cnote)
    try:
        r = requests.post(API, headers={"Authorization": f"Bearer {KEY}"},
                          json={"model": "deepseek-chat",
                                "messages": [{"role": "user", "content": body}],
                                "temperature": 1.25, "max_tokens": 500},
                          timeout=90)
        if r.status_code != 200:
            return None
        lines = [l.strip() for l in r.json()["choices"][0]["message"]["content"].splitlines()
                 if l.strip()]
    except Exception:
        return None

    parsed = []
    for l in lines:
        m = re.match(r"^\**([ABC])\**\s*[:\-]\s*(.+)$", l)
        if m:
            parsed.append((m.group(1), m.group(2).strip().strip('"')))
    if len(parsed) != len(turns):
        return None
    # speakers must follow the skeleton, or the labels would not line up
    if [p[0] for p in parsed] != [t[0] for t in turns]:
        return None
    HIN = ("bhai", "yaar", "paisa", "kitna", "acha", "theek", "haan", "nahi", "kya", "arre")
    if any(h in t.lower().split() for _, t in parsed for h in HIN):
        return None

    # The skeleton fixes the label, so the WORDING has to match the label or the row is
    # mislabelled by construction. Two ways DeepSeek drifts, both seen in the first run:
    #
    #  1. It writes a completed action on a turn meant to fire — "just sent mine too"
    #     where the skeleton said "is sending". Under FIRING_RULE 3a that must NOT fire,
    #     so the row would teach the opposite of the rule.
    #  2. It answers a money scenario with a ride, or the reverse — one m_offer_accepted
    #     came back as "could you book the uber for me? i'll venmo you".
    DONE = re.compile(r"\b(just (sent|paid|booked)|already (sent|paid|booked|did)|"
                      r"sent it|paid it|its booked|have sent|has been sent)\b", re.I)
    MONEYW = re.compile(r"\b(send|sent|pay|paid|venmo|paypal|gpay|cashapp|zelle|upi|"
                        r"transfer|rupees|dollars|bucks|\$|₹)", re.I)
    RIDEW = re.compile(r"\b(cab|uber|ola|lyft|taxi|ride|book(ing)? (me|a|an|it)|pickup|drop)\b", re.I)
    for (sp, txt), (_, _, fire) in zip(parsed, turns):
        if fire and DONE.search(txt):
            return None                                  # (1)
        if fire == ["money"] and not MONEYW.search(txt) and not MONEYW.search(parsed[0][1]):
            return None                                  # (2)
        if fire == ["ride"] and not RIDEW.search(txt) and not RIDEW.search(parsed[0][1]):
            return None
    # A money scenario whose opening line is really a ride request, or vice versa.
    opener = parsed[0][1]
    want = {t for _, _, f in turns for t in f}
    if want == {"money"} and RIDEW.search(opener) and not MONEYW.search(opener):
        return None
    if want == {"ride"} and MONEYW.search(opener) and not RIDEW.search(opener):
        return None

    return {"scenario": name, "tier": tier, "kind": kind,
            "turns": [{"sender": sp, "text": txt, "fire": list(fire)}
                      for (sp, txt), (_, _, fire) in zip(parsed, turns)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=6, help="variations per scenario")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="data/eval/behaviour_eval.json")
    a = ap.parse_args()
    if not KEY:
        sys.exit("DEEPSEEK_API_KEY not set")

    jobs = [(i, s) for s in SCENARIOS for i in range(a.per * 6)]  # 6x: the label
    # validators below reject hard, and the reject rate is very uneven across scenarios
    want = {s[0]: a.per for s in SCENARIOS}
    got = Counter()
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(build, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r and got[r["scenario"]] < want[r["scenario"]]:
                rows.append(r); got[r["scenario"]] += 1
            if n % 60 == 0:
                print(f"  {n} requested, {len(rows)} kept", flush=True)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    turns = sum(len(r["turns"]) for r in rows)
    fires = sum(1 for r in rows for t in r["turns"] if t["fire"])
    print(f"\n  wrote {len(rows)} conversations ({turns} turns, {fires} firing) -> {out}")
    print(f"    by tier : {dict(Counter(r['tier'] for r in rows))}")
    print(f"    by kind : {dict(Counter(r['kind'] for r in rows))}")
    missing = {k: want[k] - got[k] for k in want if got[k] < want[k]}
    if missing:
        print(f"    short   : {missing}")


if __name__ == "__main__":
    main()
