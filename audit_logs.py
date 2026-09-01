"""Every failure in the real logs, grouped, counted, and quoted.

Six training rounds were planned from guesses about what the model needed. This is the
opposite: read what actually happened on real traffic and let the counts decide what the
next round contains.

Uses the log's own record of what the deployed model did, so it needs no server and no
labels invented by an LLM - the classifier's score is in the file next to the message.

Three groups, each with a different fix:

  SUPPRESSED   the classifier was confident and no prompt appeared. A pipeline problem.
  MISSED       a commitment the classifier scored low. Training data.
  FALSE        a prompt on something that is not a payment or a ride. Training data.
"""
import argparse, json, re, sys, io
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent
sys.stdout = io.TextIOWrapper(open(sys.__stdout__.fileno(), "wb", closefd=False),
                              encoding="utf-8", errors="replace", line_buffering=True)
MR = ("money", "ride")

# A commitment: the speaker agrees, or says the action is happening.
COMMIT = re.compile(
    r"^(?:ok|okay|k|kk|sure|yeah|yea|ya|yep|yup|yes|fine|of course|why not|will do|"
    r"cool|bet|done deal|\U0001F44D)\b"
    r"|\b(?:sending|booking|transferring|paying|ordering)\b"
    r"|\b(?:i\s?a?m|i'?m|i\s?will|i'?ll|ill|let me|lemme)\s+(?:just\s+)?"
    r"(?:send|book|pay|transfer|get|grab|order|arrange)\b"
    r"|\bon it\b|\bi got (?:you|u|this)\b", re.I)

# Not a commitment even though it looks like one.
NOT_YET = re.compile(
    r"\b(?:tomorrow|tonight|later|next\s+week|next\s+month|by\s+(?:mid|early|friday)|"
    r"on\s+the\s+\d+|once\s+i|when\s+i|salary)\b"          # future
    r"|\b(?:just sent|already sent|already paid|sent it|booked it|cab booked)\b"  # done
    r"|\b(?:need to|have to|gotta|should i|shall i|may|might)\b", re.I)

# A request for MONEY or a RIDE specifically - not a reminder, not a calendar entry.
MR_REQUEST = re.compile(
    r"\b(?:can|could|will|would|pls|please)\b.{0,30}"
    r"\b(?:send|pay|transfer|spot|lend|book|cab|uber|ola|ride)\b"
    r"|\b(?:send|pay|transfer|spot|lend)\s+me\b"
    r"|\bbook\s+(?:me\s+)?(?:a\s+)?(?:cab|uber|ola|ride)\b"
    r"|\bgimme\s+\d|\bneed\s+\d+\s*(?:bucks|rs|rupees)", re.I)

# The team talking about the product. Three of the four confirmed false fires were
# this, and it inflated the miss count too - "Yes" answering "did you get the
# intent?" is not a payment.
ABOUT_APP = re.compile(
    r"\bintents?\b|\bprompts?\b|\bmodel\b|\bbuild\b|\bdeploy\b|\blogs?\b"
    r"|\bnotification|\btest(?:ing)?\b|\bdogfood|\bguardrail|\bthe app\b"
    r"|\bchat\s+(?:screen|window)\b|\bwifi\b|\bmobile\s+data\b", re.I)

# A request for something that is not ours. If one of these is the most recent thing
# asked, an agreement after it belongs to that, not to a cab from earlier in the room.
OTHER_REQUEST = re.compile(
    r"\bremind(?:er|\s+me)\b|\bset\s+(?:me\s+)?a?n?\s*(?:alarm|reminder|calendar)\b"
    r"|\bcalendar\b|\bmeet(?:ing)?\s+(?:tomorrow|tmrw|at|on)\b|\bcall\s+me\b"
    r"|\bsend\s+(?:the\s+)?(?:build|version|apk|link|screenshot|message|msg)\b", re.I)

# Rooms this tool created. Every replay and battery run today wrote into the live
# log, and counting my own traffic as real usage would plan the next round around
# messages I generated myself.
TEST_ROOM = re.compile(r"^(?:nw|cr|rl|rw|si|ff|re|cs|ew|bs|trace|wgp|meta|metaL|"
                       r"g15|lmb|p1|clean|fresh|final|smoke|logtest|idchk|load|"
                       r"blindproof|slotchk|cur|dm_selfack|dm_verify|group_verify)", re.I)

# Things a prompt must never appear on.
NEVER = {
    "rideshare post":  re.compile(r"\bride\s*(?:share)?\s+available|seats?\s+available|"
                                  r"\bleaving\s+(?:from|at)\b.*\bcontact\b", re.I),
    "own car / lift":  re.compile(r"\b(?:my|his|her|their)\s+car\b|dropping me|"
                                  r"\bgiving me a (?:lift|ride)\b", re.I),
    "food":            re.compile(r"\b(?:pizza|dinner|lunch|breakfast|swiggy|zomato|"
                                  r"order(?:ing)?\s+food)\b", re.I),
    "about the app":   re.compile(r"\bintent\b|\bprompt\b|\bmodel\b|\bbuild\b|"
                                  r"\bnotification\b|\btest(?:ing)?\b", re.I),
    "greeting":        re.compile(r"^(?:h+i+|h+e+y+|h+e+l+o+|y+o+|good\s+\w+)[\s!.,?]*$", re.I),
    "price talk":      re.compile(r"\b(?:expensive|costs?|price|pricey|how much (?:is|was))\b", re.I),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=0.90)
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    logs = sorted((ROOT / "data/eval").glob("dogfood_*.jsonl"))
    rows = []
    for p in logs:
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                rows.append(json.loads(l))
    # a message can appear in more than one log; key on room+ts+text
    seen, uniq = set(), []
    for r in rows:
        k = (r["room"], r["ts"], r["text"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    rows = uniq
    before = len(rows)
    rows = [r for r in rows if not TEST_ROOM.match(r["room"])]
    print(f"  dropped {before - len(rows)} messages from this tool's own test rooms")
    by = defaultdict(list)
    for r in rows:
        by[r["room"]].append(r)
    print(f"  {len(rows)} unique messages, {len(by)} rooms, from {len(logs)} logs\n")

    sup, missed, false = [], [], []
    for room, rs in by.items():
        for i, r in enumerate(rs):
            fired = [x for x in (r.get("fired") or []) if x in MR]
            c = r.get("conv") or {}
            top = max((float(c.get(x, 0) or 0) for x in MR), default=0.0)
            txt = r["text"]
            prev = rs[i - 1]["text"][:36] if i else ""

            if top >= a.min_score and not fired:
                back = next((i - j for j in range(i - 1, -1, -1)
                             if [x for x in (rs[j].get("fired") or []) if x in MR]), None)
                sup.append((room, top, back, prev, r["sender"], txt))
            if fired:
                for name, pat in NEVER.items():
                    if pat.search(txt):
                        false.append((name, room, fired[0], prev, r["sender"], txt))
                        break
            if not fired and top < 0.5 and COMMIT.search(txt) and not NOT_YET.search(txt):
                # Scoped to money and rides. The request has to be the most recent
                # thing anyone asked for - a cab from earlier in the room does not
                # get to claim a "Sure" that answers a reminder - and the agreement
                # has to arrive within 3 messages of it.
                if ABOUT_APP.search(txt):
                    continue
                asked = None
                for j in range(i - 1, max(-1, i - 4), -1):
                    other = rs[j]
                    if ABOUT_APP.search(other["text"]):
                        continue
                    if OTHER_REQUEST.search(other["text"]):
                        asked = False       # answered something that is not ours
                        break
                    if (other["sender"] != r["sender"]
                            and MR_REQUEST.search(other["text"])):
                        asked = True
                        break
                acts = (bool(re.search(r"\b(?:sending|booking|transferring)\b", txt, re.I))
                        and not ABOUT_APP.search(txt))
                if asked is True or acts:
                    missed.append((room, top, prev, r["sender"], txt))

    print(f"  === 1. SUPPRESSED - model sure, no prompt ({len(sup)}) ===")
    print(f"      pipeline problem, split by distance from the previous prompt\n")
    near = [x for x in sup if x[2] is not None and x[2] <= 6]
    far = [x for x in sup if x[2] is None or x[2] > 6]
    print(f"      within 6 messages (echo, correct): {len(near)}")
    print(f"      further out or no prior fire      : {len(far)}")
    for room, top, back, prev, snd, txt in far[:a.show]:
        print(f"        [{room}] {top:.3f} {back if back else '-'} back | {snd}: {txt[:52]!r}")

    print(f"\n  === 2. MISSED - a commitment the model scored low ({len(missed)}) ===")
    print(f"      training data\n")
    for room, top, prev, snd, txt in missed[:a.show * 2]:
        print(f"        [{room}] {top:.3f} after {prev!r}")
        print(f"           {snd}: {txt[:56]!r}")

    print(f"\n  === 3. FALSE - a prompt on something that is not money or a ride ({len(false)}) ===")
    print(f"      training data\n")
    for name, cnt in Counter(x[0] for x in false).most_common():
        print(f"        {name:16} {cnt}")
    for name, room, intent, prev, snd, txt in false[:a.show]:
        print(f"        [{name}] {intent} | {snd}: {txt[:52]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
