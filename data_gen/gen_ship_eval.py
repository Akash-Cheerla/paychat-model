"""Fresh conversations for the ship/no-ship call — generated AFTER the fixes, so no
part of it has been seen by any model or used to tune any threshold.

This is an EVAL set, not training data. Two rules follow from that:

  * Nothing here is merged into any conversations_v*.json. The contamination check in
    merge_v6.py reads data/eval/, so writing here is what keeps it honest.
  * The labels come from derive() — the same function build_conv_dataset.py uses. Roles
    invented by hand are how 997 offer labels silently became 0 in the v6 round, so the
    role vocabulary below is asserted against derive() before anything is written.

Shapes are the ones that actually broke in real chats, plus the model gaps measured on
2026-08-10. English only — the India market types in English in this product, and every
earlier round that mixed in Hinglish had to be thrown away.

Run:  DEEPSEEK_API_KEY=... python data_gen/gen_ship_eval.py --n 400 --workers 8
"""
import argparse, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
API = "https://api.deepseek.com/chat/completions"
KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OUT = ROOT / "data/eval/ship_eval.json"

# Every scenario states the role of the LAST turn, because that is the turn the
# firing decision is about. The generator writes the surrounding chat; it never
# chooses the label.
# (name, kind, last_role, request_role, description)
#
# request_role is what makes this work. derive() only fires an acknowledgement if a
# REQUEST is already open — an ack with nothing to answer derives to nothing. Tagging
# only the final turn produced 1 firing turn in 247, a set that would have scored as
# flawless while testing nothing. So the generator marks the request line too, and the
# roles are attached to marked lines rather than guessed.
SCENARIOS = [
    # ── the shapes that broke in real chats ──
    ("ride_addr_lines", "group", "ack_ride", "request_ride",
     "Someone asks for a cab and gives their pickup address on its own line, the way "
     "people actually paste an address. Another person agrees to book it."),
    ("ride_new_trip", "dm", "ack_ride", "request_ride",
     "A cab is booked and settled. Later the SAME pair arrange a completely different "
     "trip, different places. The last turn agrees to book the new one."),
    ("ride_rider_only_dest", "dm", "ack_ride", "request_ride",
     "A asks for a cab to a place. B suggests a different destination. A never agrees. "
     "B books it anyway on the last turn."),
    ("money_counter", "dm", "ack_money", "request_money",
     "A asks to borrow an amount, B offers a smaller one, A accepts the smaller one on "
     "the last turn."),
    ("money_split_known", "group", "ack_money", "request_money",
     "A bill is split a stated number of ways among the people in the chat. The last "
     "turn is one person paying their share."),

    # ── the model gaps measured 2026-08-10 ──
    ("ride_self_initiated", "group", "self_ride", None,
     "Nobody asks for anything. Someone announces they are booking a cab for the group "
     "and names where to. The announcement is the last turn."),
    ("ride_substitute", "group", "ack_ride", "request_ride",
     "A cab is needed and one person volunteers. A DIFFERENT person then takes over the "
     "booking on the last turn, and the first person steps aside."),
    ("money_offer_accepted", "dm", "ack_money", "request_money",
     "A OFFERS to send B money (B never asked). B accepts on the last turn."),

    # ── must-not-fire ──
    ("neg_friend_lift", "dm", "neutral", None,
     "A asks a friend for a lift in the friend's OWN car. The friend agrees. No cab, "
     "taxi, Uber or booking app is mentioned by anyone."),
    ("neg_completed", "dm", "already_done", "request_money",
     "Someone asks whether a payment was made and the other person says it was already "
     "sent earlier. Nothing is owed now."),
    ("neg_rejected", "dm", "reject", "request_ride",
     "Someone asks for a cab to be booked and the other person refuses on the last turn."),
    ("neg_discussion", "group", "neutral", None,
     "People discuss the cost of a trip or a meal without anyone asking anyone to pay "
     "or to book anything. Amounts are mentioned."),
    ("neg_future", "dm", "future_promise", "request_money",
     "Someone is asked for money and promises to send it at a stated later time, "
     "without sending it now."),
]

HIN = ("bhai", "yaar", "paisa", "kitna", "acha", "theek", "haan", "nahi", "kya", "arre",
       "chalo", "matlab", "bolo", "karo")

PROMPT = """Write a realistic {kind} chat message thread in ENGLISH between {who}.

Situation: {desc}

Rules:
- {n} messages total, ending exactly on the turn described above.
- Format each line as "A: text" using only the letters {letters}. Nothing else.
- Lowercase, typos, short messages. Real texting, not clean prose.
- NO Hindi or Hinglish words at all. English only.
- Do not label, explain or number anything. Just the messages.
{marks}"""

MARK_REQ = ("- Put [R] at the very start of the ONE message where someone asks for the "
            "money or the cab (after the speaker letter). Exactly one line gets [R].\n")
MARK_LAST = ("- Put [C] at the very start of the FINAL message (after the speaker "
             "letter).\n")
# Without this, the model writes the commitment as an accomplished fact ("booked it!",
# "sent!"), which FIRING_RULE §3a correctly derives to no-fire — so a whole positive
# scenario silently became a negative one and tested nothing it was written for.
MARK_NOT_DONE = ("- In that final message the person AGREES to do it or is doing it "
                 "right now. They must NOT say it is already finished.\n")


def call(body, temp=1.25):
    try:
        r = requests.post(API, headers={"Authorization": f"Bearer {KEY}"},
                          json={"model": "deepseek-chat",
                                "messages": [{"role": "user", "content": body}],
                                "temperature": temp, "max_tokens": 700},
                          timeout=90)
        if r.status_code != 200:
            return None
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def parse(txt, speakers):
    out = []
    for line in (txt or "").splitlines():
        m = re.match(r"^\**([A-Z])\**\s*[:\-]\s*(.+)$", line.strip())
        if m and m.group(1) in speakers:
            out.append((m.group(1), m.group(2).strip().strip('"')))
    return out


def one(scen):
    name, kind, last_role, req_role, desc = scen
    speakers = ["A", "B"] if kind == "dm" else \
        ["A", "B", "C"] + (["D"] if random.random() < 0.35 else [])
    n = random.randint(5, 12) if kind == "dm" else random.randint(6, 14)
    who = "two friends" if kind == "dm" else f"{len(speakers)} friends in a group chat"
    marks = ((MARK_REQ if req_role else "") + MARK_LAST
             + (MARK_NOT_DONE if last_role in ("ack_money", "ack_ride", "self_ride") else ""))
    txt = call(PROMPT.format(kind="direct message" if kind == "dm" else "group",
                             who=who, desc=desc, n=n, letters=", ".join(speakers),
                             marks=marks))
    seq = parse(txt, set(speakers))
    if len(seq) < 4:
        return None
    if any(h in " ".join(t.lower() for _, t in seq) for h in HIN):
        return None                                   # English only, no exceptions

    turns, req_at, last_at = [], None, None
    for i, (s, t) in enumerate(seq):
        role = "neutral"
        if t.startswith("[R]") and req_role and req_at is None:
            role, req_at = req_role, i
        elif t.startswith("[C]"):
            role, last_at = last_role, i
        turns.append({"sender": s, "text": re.sub(r"^\[[RC]\]\s*", "", t), "role": role})

    # A conversation missing its markers is unlabelled, not neutral. Keeping it would
    # quietly add a "nothing fires here" row to the positives.
    if req_role and req_at is None:
        return None
    if last_at is None:
        turns[-1]["role"] = last_role
        last_at = len(turns) - 1
    if req_at is not None and req_at >= last_at:
        return None                                   # the ack must follow the request
    if req_at is not None and turns[req_at]["sender"] == turns[last_at]["sender"]:
        return None                                   # nobody answers their own request
    return {"scenario": name, "kind": kind, "turns": turns}


# What each role MUST derive to, in a minimal conversation. This is the v6 disaster
# check, and it has to be written this way: derive() returns {**turn, "fire": ...}, so
# it echoes whatever role you hand it. Comparing roles in against roles out therefore
# always passes, including for a role derive() has never heard of — which is exactly
# how an invented role produced 997 silently-zeroed labels. The only honest question is
# what derive() DOES with the role, so every role is probed for its fire outcome.
ROLE_FIRES = {
    "ack_money": ["money"], "ack_ride": ["ride"],
    "self_ride": ["ride"], "self_money": ["money"],
    # A request opens a pending and fires nothing by itself — FIRING_RULE §1. The
    # prompt appears when someone commits, which is the whole premise of the product.
    "request_money": [], "request_ride": [],
    # An offer is a request with the direction reversed: it opens a pending, fires
    # nothing itself, and the acceptance fires. Added 2026-08-11 — before that an offer
    # was labelled request_money, which hid it inside a role scoring 99.8% while the
    # coverage gate measured offers failing 40-60% of the time.
    "offer_money": [], "offer_ride": [],
    # §1a — a statement creates no pending, so a bare ack has nothing to accept.
    "statement_money": [], "statement_ride": [],
    "neutral": [], "reject": [], "future_promise": [], "already_done": [],
}
# A role that fires needs an open request to answer; a self-initiated one does not.
NEEDS_REQUEST = {"ack_money": "request_money", "ack_ride": "request_ride"}


def assert_roles(rows):
    """Refuse to write unless derive() actually produces the intended labels."""
    from gen_conversations import derive
    used = sorted({t["role"] for c in rows for t in c["turns"]})
    unspecified = [r for r in used if r not in ROLE_FIRES]
    if unspecified:
        sys.exit(f"REFUSING TO WRITE — no expected outcome declared for {unspecified}. "
                 f"Add it to ROLE_FIRES and confirm against FIRING_RULE.md.")

    bad = []
    for role in used:
        seq = []
        if role in NEEDS_REQUEST:
            seq.append({"sender": "A", "text": "can you sort this out",
                        "role": NEEDS_REQUEST[role]})
        seq.append({"sender": "B", "text": "ok doing that", "role": role})
        got = derive(seq)[-1].get("fire") or []
        if sorted(got) != sorted(ROLE_FIRES[role]):
            bad.append(f"{role}: derive() gave {got}, expected {ROLE_FIRES[role]}")
    if bad:
        sys.exit("REFUSING TO WRITE — derive() disagrees with the intended labels:\n  "
                 + "\n  ".join(bad))

    fired = sum(1 for c in rows for t in derive(c["turns"]) if t.get("fire"))
    total = sum(len(c["turns"]) for c in rows)
    print(f"derive() cross-check: {len(used)} roles verified, {fired}/{total} firing turns")
    if not fired:
        sys.exit("REFUSING TO WRITE — derive() produced zero firing turns")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not KEY:
        sys.exit("DEEPSEEK_API_KEY not set")

    jobs = [SCENARIOS[i % len(SCENARIOS)] for i in range(a.n)]
    rows, dropped = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, s) for s in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                rows.append(r)
            else:
                dropped += 1
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}  kept {len(rows)}  dropped {dropped}")

    seen, uniq = set(), []
    for c in rows:
        k = "|".join(t["text"] for t in c["turns"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    print(f"\nkept {len(uniq)} unique ({len(rows) - len(uniq)} duplicates, {dropped} dropped)")

    assert_roles(uniq)

    # Write the DERIVED fire label onto every turn. run_eval_server.py reads "fire"
    # (or "expected"); a file carrying only roles evaluates as "nothing should ever
    # fire", which reads as a flawless score on a set that tests nothing.
    from gen_conversations import derive
    for c in uniq:
        c["turns"] = [{"sender": t["sender"], "text": t["text"], "role": t["role"],
                       "fire": t.get("fire") or []}
                      for t in derive(c["turns"])]

    from collections import Counter
    print("by scenario:", dict(Counter(c["scenario"] for c in uniq)))
    print("group share:", sum(1 for c in uniq if c["kind"] == "group"), "/", len(uniq))
    print("firing turns:", sum(1 for c in uniq for t in c["turns"] if t["fire"]),
          "of", sum(len(c["turns"]) for c in uniq))
    OUT.write_text(json.dumps(uniq, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
