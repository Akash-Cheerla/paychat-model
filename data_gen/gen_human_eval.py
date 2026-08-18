"""Generate a full-coverage natural-conversation test set for money and ride.

Why this exists. Gowtham's bar is "the two intents work flawlessly, 90+% with human
conversation". We could not measure that. Every eval we owned was 99.6% money/ride
adjacent, so none of them could see the failure that actually reaches users: a payment
sheet opening because somebody typed "Okay!".

The first cut of this file was DM-only and had no hard cases. This one covers the
matrix properly:

    DM and GROUP  x  SHOULD-FIRE and SHOULD-STAY-QUIET  x  SIMPLE and HARD

Every scenario is a case we have actually seen fail, or a case the firing rule calls
out. The label comes from the scenario design, never from a model guessing afterwards,
and the quiet conversations are screened for stray commitments before being written.

Style constraints come from measuring real chats (317k lines of WhatsApp): median 3-7
words per message, half of them 3 words or fewer, intent buried in ordinary talk rather
than announced in the first two lines.

    DEEPSEEK_API_KEY=... python data_gen/gen_human_eval.py --n 1200 --workers 14
    python data_gen/gen_human_eval.py --audit 30
"""
import argparse, json, os, random, re, sys, io, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
API = "https://api.deepseek.com/chat/completions"
KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OUT = ROOT / "data/eval/human_eval.json"

STYLE_DM = """Write it like real texting between two friends, not like a script:
- most messages 2-6 words, many just one or two
- lowercase mostly, occasional typo, no perfect punctuation
- people change subject, reply late, send two messages in a row
- no narration, no stage directions
Format every line as "A: text" or "B: text". Nothing else."""

STYLE_GROUP = """Write it like a real group chat with {k} people, not like a script:
- most messages 2-6 words, many just one or two
- lowercase mostly, occasional typo
- several people talk at once, some threads get ignored, side jokes happen
- no narration, no stage directions
Format every line as "A: text", "B: text", "C: text"{extra}. Nothing else."""

# ---------------------------------------------------------------------------
# SHOULD FIRE. (name, brief, intent, tier)
FIRE_DM = [
    ("m_request_ack", "A asks B for {amt} for a real reason. A few unrelated messages "
     "happen. Then B clearly agrees to send it now.", "money", "simple"),
    ("m_self_initiated", "A and B chat about ordinary things. Then A remembers they owe "
     "B {amt} and says they are sending it right now.", "money", "simple"),
    ("r_request_ack", "A asks B to book them a cab to {dst}. A few unrelated messages "
     "happen. Then B agrees and says they are booking it.", "ride", "simple"),
    ("r_self_initiated", "A and B chat about their day. Then A says they are booking a "
     "cab to {dst} right now for B, without being asked.", "ride", "simple"),
    ("m_offer_then_handle", "B notices A is short of cash and offers to cover {amt}. "
     "A accepts. B then asks A for their payment handle so they can send it.",
     "money", "hard"),
    ("m_counter_offer", "A asks B for {amt}. B says they can only manage a smaller "
     "amount. A accepts the smaller amount. B then sends that smaller amount now.",
     "money", "hard"),
    ("m_after_interruption", "A asks B for {amt}. Then the subject changes completely "
     "for six or seven messages - work, a match, family. Only after all that does B "
     "come back and agree to send it now.", "money", "hard"),
    ("r_after_interruption", "A asks B to book a cab to {dst}. The subject changes "
     "completely for six or seven messages. Only after that does B agree and book it.",
     "ride", "hard"),
    ("m_chased_then_pays", "B has been chasing A about {amt} for days. They argue about "
     "it a bit. Finally A gives in and sends it right now.", "money", "hard"),
    ("m_conditional_then_yes", "A asks B for {amt}. B first says maybe, they need to "
     "check their balance. They talk about something else. Then B checks and confirms "
     "they are sending it now.", "money", "hard"),
    ("r_reluctant_yes", "A asks B to book a cab to {dst}. B complains about the cost "
     "and the time, they go back and forth, but B eventually agrees and books it.",
     "ride", "hard"),
]

FIRE_GROUP = [
    ("g_m_request_ack", "A asks the group for {amt} for a shared expense. Others chat. "
     "Then C agrees to send it now.", "money", "simple"),
    ("g_r_request_ack", "A asks the group if anyone can book a cab to {dst}. Others "
     "chat about something else. Then B agrees and books it.", "ride", "simple"),
    ("g_someone_else_takes_it", "A asks the group for a cab to {dst}. B says they "
     "cannot. C then takes it on and books it.", "ride", "hard"),
    ("g_split_then_one_pays", "The group works out splitting a bill of {amt}. There is "
     "some argument about the split. Then C says they are sending their share now.",
     "money", "hard"),
    ("g_two_requests_one_ack", "A asks for money for a shared bill. Separately B asks "
     "for a cab to {dst}. Then C clearly answers only the CAB request and books it.",
     "ride", "hard"),
]

# ---------------------------------------------------------------------------
# SHOULD STAY QUIET. (name, brief, tier)
QUIET_DM = [
    ("chit_chat", "Two friends have an ordinary conversation - a match, a show, family, "
     "weather, being tired. No money and no travel at all.", "simple"),
    ("money_talk_no_transfer", "Two friends complain about prices - rent going up, a "
     "{amt} electricity bill, how expensive something was. Nobody asks for or sends "
     "any money.", "simple"),
    ("public_transport", "Two friends talk about catching a bus or train somewhere, "
     "times and delays. No cab or taxi is booked.", "simple"),
    ("work_ack", "Two colleagues discuss a work task - a deploy, a document, an email, "
     "a bug. One agrees to do it and says things like 'on it', 'doing it now', 'sure', "
     "'okay!'. NO money and NO cab anywhere.", "hard"),
    ("send_non_money", "Two people talk about SENDING something that is not money - a "
     "file, a link, an email, a photo, an address. Someone says they are sending it "
     "now.", "hard"),
    ("friend_drives", "A asks B for a lift. B agrees to drive them in B's own car and "
     "they sort out the time and pickup spot. No cab, taxi, uber or ola is ever "
     "mentioned.", "hard"),
    ("food_order", "Friends decide what food to order and who is ordering it. One says "
     "they are ordering now. No money moves between them and no cab is involved.",
     "hard"),
    ("booking_not_a_ride", "Two friends book something that is NOT a cab - a restaurant "
     "table, a hotel, a doctor appointment, cinema tickets. One says they are booking "
     "it now.", "hard"),
    ("already_paid", "A and B talk about a bill A already settled days ago. A mentions "
     "having paid it. Nothing further is owed and nobody sends anything.", "hard"),
    ("future_promise", "A owes B {amt}. A says they will pay it back next week when "
     "they get paid. Nobody sends anything now.", "hard"),
    ("rejection", "A asks B for {amt}. B says no, they cannot do it right now. A "
     "accepts that. Nobody sends anything.", "hard"),
    ("ride_rejection", "A asks B to book a cab to {dst}. B refuses - too expensive, or "
     "they are busy. A drops it. No cab is ever booked.", "hard"),
    ("preparation_only", "A asks B for {amt}. B says they are opening the app and "
     "looking for it, then gets distracted by something else and never actually "
     "confirms sending anything.", "hard"),
    ("discussed_never_agreed", "Two friends discuss splitting a {amt} cost and go back "
     "and forth about who owes what. They never settle it and nobody sends anything.",
     "hard"),
    ("cash_offline", "A owes B {amt}. They agree A will hand over cash in person "
     "tomorrow. No app, no transfer, nothing sent now.", "hard"),
    ("third_party_payment", "A and B talk about someone ELSE paying for something - a "
     "company reimbursing, a parent paying a fee, an insurance claim. Neither of them "
     "sends anything to the other.", "hard"),
]

QUIET_GROUP = [
    ("g_chit_chat", "A group of friends chat about a match, weekend plans and food. "
     "No money and no cabs.", "simple"),
    ("g_nobody_takes_it", "A asks the group for a cab to {dst}. Everyone is busy or "
     "makes excuses. Nobody books anything.", "hard"),
    ("g_plan_only", "A group plans a trip - who is coming, what time, what to bring. "
     "Travel is discussed but nobody books a cab and no money moves.", "hard"),
    ("g_food_order", "A group decides what to order for dinner and who will place the "
     "order. One says they are ordering now. No money moves between them.", "hard"),
    ("g_bill_discussion", "A group works out who owes what for a {amt} bill, arguing "
     "about the split. They agree to settle it later. Nobody sends anything now.",
     "hard"),
]

AMT = ["500", "20", "1500", "300", "40", "2000", "250", "80", "100", "60", "750"]
DST = ["koramangala", "the airport", "hsr layout", "indiranagar", "jp nagar",
       "the station", "whitefield", "electronic city", "the mall", "btm layout"]


def ask(prompt, tries=3):
    for a in range(tries):
        try:
            r = requests.post(API, timeout=180,
                              headers={"Authorization": f"Bearer {KEY}"},
                              json={"model": "deepseek-chat", "temperature": 1.25,
                                    "messages": [{"role": "user", "content": prompt}]})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        time.sleep(2 * (a + 1))
    return ""


LINE = re.compile(r"^\s*([A-E])\s*:\s*(.+?)\s*$")

# A turn that commits to paying or booking RIGHT NOW. Polices the QUIET scenarios: the
# brief says nobody sends money or books a cab, but the model writes freely for 20+
# turns and can wander into a real commitment anyway. Such a conversation is
# mislabelled by construction, so it is dropped rather than shipped with a label we
# already know is wrong.
COMMIT = re.compile(
    r"\b(sending\s+(?:it|you|u|now|the\s+money)|i'?ll\s+send\s+(?:it|you|u)\s*(?:now|rn)|"
    r"transferr?ing\s+(?:it|now)|paying\s+(?:you|u)\s+now|sent\s+it\s+now|"
    r"book(?:ing)?\s+(?:it|one|the\s+cab|an?\s+(?:uber|ola|cab|lyft))|"
    r"ordering\s+(?:you\s+)?an?\s+(?:cab|uber|ola))\b", re.IGNORECASE)
PAYWORD = re.compile(r"\b(venmo|zelle|cashapp|gpay|upi|paytm|paypal|transfer|"
                     r"\$\d|\d+\s?(?:rs|inr|usd|cad|bucks|dollars))\b", re.IGNORECASE)
CABWORD = re.compile(r"\b(cab|taxi|uber|ola|lyft|rapido)\b", re.IGNORECASE)


def stray_commitment(turns):
    for t in turns:
        if COMMIT.search(t["text"]) and (PAYWORD.search(t["text"])
                                         or CABWORD.search(t["text"])):
            return t["text"]
    return None


def to_turns(raw, allowed):
    out = []
    for ln in (raw or "").splitlines():
        m = LINE.match(ln)
        if m and m.group(1) in allowed:
            t = m.group(2).strip()
            if t and not t.startswith(("(", "[", "*")):
                out.append({"sender": m.group(1), "text": t})
    return out


def one(rng):
    group = rng.random() < 0.35
    fires = rng.random() < 0.45
    if group:
        pool = FIRE_GROUP if fires else QUIET_GROUP
        k = rng.choice([3, 3, 4, 5])
        allowed = "ABCDE"[:k]
        style = STYLE_GROUP.format(
            k=k, extra="".join(f', "{c}: text"' for c in allowed[3:]))
    else:
        pool = FIRE_DM if fires else QUIET_DM
        allowed = "AB"
        style = STYLE_DM

    if fires:
        name, brief, intent, tier = rng.choice(pool)
    else:
        name, brief, tier = rng.choice(pool)
        intent = None

    brief = brief.format(amt=rng.choice(AMT), dst=rng.choice(DST))
    n = rng.choice([10, 12, 14, 16, 18])
    tail = ("The moment someone commits to doing it must appear, and ordinary chat "
            "should continue for a few messages afterwards."
            if fires else
            "Nobody in this conversation ever sends money or books a cab.")
    raw = ask(f"{brief}\n\nWrite about {n} messages.\n{tail}\n\n{style}")
    turns = to_turns(raw, set(allowed))
    if len(turns) < 6:
        return None
    turns = turns[:24]
    for t in turns:
        t["fire"] = []

    if fires:
        # Mark the LAST turn that reads as a commitment; the scorer accepts a fire
        # anywhere, so this is a hint for auditing rather than a hard position.
        idx = next((i for i in range(len(turns) - 1, -1, -1)
                    if COMMIT.search(turns[i]["text"])), len(turns) - 1)
        turns[idx]["fire"] = [intent]
    else:
        stray = stray_commitment(turns)
        if stray:
            return {"_rejected": stray, "scenario": name}

    return {"scenario": name, "expects_fire": bool(fires), "tier": tier,
            "kind": "group" if group else "DM", "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--audit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    if a.audit:
        d = json.loads(OUT.read_text(encoding="utf-8"))
        for c in random.Random(99).sample(d, min(a.audit, len(d))):
            print(f"\n--- {c['scenario']} [{c['kind']}/{c['tier']}] "
                  f"expects_fire={c['expects_fire']}")
            for t in c["turns"]:
                mark = "   <<< FIRE " + str(t["fire"]) if t["fire"] else ""
                print(f"    {t['sender']}: {t['text'][:68]}{mark}")
        return 0

    if not KEY:
        sys.exit("DEEPSEEK_API_KEY not set")
    seeds = [random.Random(a.seed + i) for i in range(a.n)]
    out, rejected = [], []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, s) for s in seeds]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                c = f.result()
            except Exception:
                c = None
            if c and c.get("_rejected"):
                rejected.append(c)
            elif c:
                out.append(c)
            if k % 100 == 0:
                print(f"  {k}/{a.n}  kept {len(out)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    fire = sum(1 for c in out if c["expects_fire"])
    grp = sum(1 for c in out if c["kind"] == "group")
    hard = sum(1 for c in out if c["tier"] == "hard")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(out)} conversations, {sum(len(c['turns']) for c in out)} turns")
    print(f"    fire {fire}  quiet {len(out)-fire}")
    print(f"    DM {len(out)-grp}  group {grp}")
    print(f"    simple {len(out)-hard}  hard {hard}")
    print(f"  dropped {len(rejected)} quiet conversations that wandered into a real "
          f"commitment")
    for k, v in Counter(c["scenario"] for c in out).most_common():
        print(f"    {k:26} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
