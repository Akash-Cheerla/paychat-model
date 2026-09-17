"""Find (and optionally apply) training labels that contradict Akash's 2026-09-15 rulings.

R1  agreement + a detail question after someone else's request ("sure, how much?",
    "sure where u going?") is a commitment and fires. The generators labelled these
    "question" and quiet.
R2  the same request re-sent after it was accepted, then accepted again: the second
    acceptance is quiet, unless someone said the attempt failed in between.
R3  a group where a second person accepts the same request: fires, unless someone
    confirmed it done in between.

Detection is deliberately narrow - a window is only touched when the pattern is
unambiguous - and prints samples so every rule can be read before it is applied.

    python data_gen/rulings_0915_relabel.py conv_windows_v16.zip            # report only
    python data_gen/rulings_0915_relabel.py conv_windows_v16.zip --apply OUT # write
"""
import argparse, io, json, random, re, sys, zipfile
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent

MONEY_KW = re.compile(r"\b(send|pay|lend|spot|venmo|paytm|gpay|upi|zelle|transfer|wire|borrow|cover|money|cash|bucks|dollars|rupees|rs|inr)\b|[$₹]", re.I)
RIDE_KW = re.compile(r"\b(cab|uber|ola|lyft|taxi|ride|rapido|auto)\b", re.I)
REQUEST_SHAPE = re.compile(r"\b(can|could|would|will|pls|please)\b.*\b(you|u|someone|anyone)\b|\b(send|book|get|pay|lend|spot|venmo|order|call|paytm|gpay)\s+(me|us)\b|\bneed\s+(a|some|\d)", re.I)
AGREE_Q = re.compile(r"^\W*(sure|ok|okay|yeah|yes|yep|ya|yup|alright|of course|definitely)\b[\s,.!-]*"
                     r"(?=.*\?|.*\b(how much|where|what time|when|which|pickup|pick you up|address)\b)", re.I)
NEGATING = re.compile(r"\b(but|can'?t|cannot|not|no|nah|later|tomorrow|friday|next week|busy|broke|maybe)\b", re.I)
CANCEL = re.compile(r"\b(nvm|never ?mind|cancel|no need|forget it|actually no|all good now|sorted)\b", re.I)
ACK = re.compile(r"^\W*(ok|okay|k|kk|sure|yes|yep|yeah|ya|yup|bet|on it|got you|i got you|sure thing|i can send too|me too|sending( too| mine)?|ok sending)\W*$", re.I)
DONE = re.compile(r"\b(sent|paid|done|booked|transferred)\b", re.I)
FAILED = re.compile(r"\b(did\s*n[o']?t\s+get|didn'?t\s+get|no\s+intent|not\s+getting|didn'?t\s+work|not\s+working|didn'?t\s+go\s+through|failed|try\s+again)\b", re.I)


def tokens(t):
    return set(re.findall(r"[a-z0-9]+", (t or "").lower()))


# Agreement first, then a question about a detail OF THE REQUEST. The first version only
# accepted the question immediately after the agreement token, so "sure i got you, gpay or
# paytm?" and "sure bro, amy at 6?" stayed quiet and kept teaching the opposite of ruling 1
# - which is why 1,518 new windows could not move that family in the tuning probe.
_AGREE_HEAD = r"^\W*(sure|ok|okay|yeah|yes|yep|ya|yup|alright|of course|definitely|k|kk)\b[^?]{0,40}"
_DETAIL_Q = (r"(how much|how many|what'?s the (amount|address|total|location|destination)|amount\?|total\?|"
             r"what time|when do you (need|want)|by when|where (to|are you|should i|do you)|which "
             r"(terminal|area|location|place|one|app)|pickup|pick (you|u) up|how do (you|u) want it|"
             r"(cash ?app|venmo|gpay|paytm|phonepe|upi|paypal|zelle|cash|card|uber|ola|rapido)\s*(or|/)\s*"
             r"(cash ?app|venmo|gpay|paytm|phonepe|upi|paypal|zelle|cash|card|uber|ola|rapido)|"
             r"\bat \d|\b\d{1,2}(:\d{2})?\s*(am|pm)\b)")
AGREE_DETAIL_Q = re.compile(_AGREE_HEAD + _DETAIL_Q, re.I)
NOT_R1 = re.compile(r"\b(sent|booked|paid|done|transferred|did you|have you|what was|but|can'?t|not|no|nah|later|busy|broke)\b", re.I)
# FIRING_RULE s4: a friend's own car is not ride-hailing. s3c: a need is not a request.
OWN_CAR = re.compile(r"\b(drive|drop|pick)\s+(me|us|u|you)\b|\b(give|gimme)\s*(me|us|u)?\s*a?\s*(ride|lift)\b|"
                     r"\blift\b|\bi'?ll\s+drive\b|\bmy\s+car\b", re.I)
NEED_ONLY = re.compile(r"^\W*((hey|hi|also|so|yo|btw|and)[\s,]+){0,3}(i|we)\s+(need|have)\s+to\b", re.I)
# a money verb or an amount outranks a ride noun: "need 300 for uber" is a money request
MONEY_ACT = re.compile(r"\b(send|pay|lend|spot|venmo|paytm|gpay|upi|zelle|transfer|wire|borrow)\b|[$₹]|\b\d{2,}\b", re.I)
# stricter than MONEY_KW: "cover for me at the meeting" is not money
MONEY_STRICT = re.compile(r"\b(send|pay|lend|spot|venmo|paytm|gpay|upi|zelle|transfer|wire|borrow|money|cash|bucks|dollars|rupees|rs|inr)\b|[$₹]", re.I)


def r1(w):
    """Narrow on purpose: short reply, agreement FIRST, then a question about the request's
    own details, answering a request that is the other person's latest message(s)."""
    win, last = w["window"], w["window"][-1]
    t = last["t"]
    if (w["fire"] or len(t.split()) > 12 or not AGREE_DETAIL_Q.search(t) or NOT_R1.search(t)
            or OWN_CAR.search(t)):        # "sure ill drive u" is a friend's car, FIRING_RULE s4
        return None
    i = len(win) - 2
    while i >= 0 and win[i]["s"] == last["s"]:
        return None                      # the replier spoke in between - not a direct answer
    if i < 0:
        return None
    y = win[i]["s"]
    while i >= 0 and win[i]["s"] == y:   # the requester's latest run of messages
        m = win[i]
        if CANCEL.search(m["t"]):
            return None
        if REQUEST_SHAPE.search(m["t"]):
            if OWN_CAR.search(m["t"]) or NEED_ONLY.search(m["t"]):
                return None
            booking = bool(re.search(r"\b(book|order|call|grab|get)\b", m["t"], re.I))
            if MONEY_ACT.search(m["t"]) and not booking:
                return ["money"]
            if RIDE_KW.search(m["t"]) or re.search(r"\bbook\b", m["t"], re.I):
                return ["ride"]
            if MONEY_STRICT.search(m["t"]):
                return ["money"]
            return None
        i -= 1
    return None


def r2(w):
    win, last = w["window"], w["window"][-1]
    if not w["fire"] or not ACK.search(last["t"]):
        return None
    x = last["s"]
    reqs = [(i, m) for i, m in enumerate(win[:-1]) if m["s"] != x and REQUEST_SHAPE.search(m["t"])]
    if len(reqs) < 2:
        return None
    (i1, a), (i2, b) = reqs[-2], reqs[-1]
    if a["s"] != b["s"]:
        return None
    ta, tb = tokens(a["t"]), tokens(b["t"])
    if not ta or len(ta & tb) / len(ta | tb) < 0.7 or set(re.findall(r"\d+", a["t"])) != set(re.findall(r"\d+", b["t"])):
        return None
    between = win[i1 + 1:i2]
    if not any(m["s"] == x and ACK.search(m["t"]) for m in between):
        return None
    if any(FAILED.search(m["t"]) for m in win[i1 + 1:]):
        return None
    return []


def r3(w):
    win, last = w["window"], w["window"][-1]
    if w["fire"] or len({m["s"] for m in win}) < 3 or not ACK.search(last["t"]):
        return None
    z = last["s"]
    for i in range(len(win) - 2, -1, -1):
        m = win[i]
        if m["s"] not in (z,) and REQUEST_SHAPE.search(m["t"]):
            y = m["s"]
            after = win[i + 1:-1]
            acks = [a for a in after if a["s"] not in (y, z) and ACK.search(a["t"])]
            if not acks or any(DONE.search(a["t"]) for a in after) or any(CANCEL.search(a["t"]) for a in after):
                return None
            if any(a["s"] == z for a in after):
                return None
            mo, ri = bool(MONEY_KW.search(m["t"])), bool(RIDE_KW.search(m["t"]))
            return (["money"] if mo else ["ride"]) if mo != ri else None
    return None


# R3 is NOT applied: on v16 its 11 hits were mostly two DIFFERENT requests each getting an
# "ok", not a second answer to one request. New windows teach it instead (build_conv_v17).
RULES = {"R1 agreement + question fires": r1, "R2 re-sent request, 2nd ack quiet": r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--apply", default=None, help="write a relabelled zip here")
    ap.add_argument("--samples", type=int, default=12)
    a = ap.parse_args()
    z = zipfile.ZipFile(ROOT / a.src)
    out = {}
    rnd = random.Random(15)
    for split in ("train.json", "val.json", "test.json"):
        rows = json.loads(z.read(split))
        hits = {k: [] for k in RULES}
        for w in rows:
            for name, fn in RULES.items():
                new = fn(w)
                if new is not None and new != list(w["fire"] or []):
                    hits[name].append((w, new))
                    break
        print(f"\n  {split}: {len(rows):,} windows")
        for name, h in hits.items():
            roles = Counter(w.get("role") for w, _ in h).most_common(5)
            print(f"    {name:40} {len(h):6,}   roles {roles}")
            if split == "train.json":
                for w, new in rnd.sample(h, min(a.samples, len(h))):
                    print(f"        {w['fire']} -> {new}   " + "  |  ".join(f"{m['s']}: {m['t'][:55]}" for m in w["window"][-4:]))
        if a.apply:
            for name, h in hits.items():
                for w, new in h:
                    w["fire"] = new
                    w["relabel_0915"] = name.split()[0]
        out[split] = rows
    if a.apply:
        with zipfile.ZipFile(ROOT / a.apply, "w", zipfile.ZIP_DEFLATED) as zo:
            for split, rows in out.items():
                zo.writestr(split, json.dumps(rows, ensure_ascii=False))
        print(f"\n  wrote {a.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
