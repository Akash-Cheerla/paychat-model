"""Find (and optionally apply) training labels that contradict the 2026-09-17 rulings.

R6  group: B clearly TOOK the request ("let me book a cab", "i'll send it", "sending now"),
    then a third person's BARE ok/sure -> quiet (Akash 09-17, rulings 6 and 7).
R7  an offer-shaped reply that ANSWERS a request ("can you book me a cab" / "let me book a
    cab") is a commitment and fires (FIRING_RULE s3c, Gowtham 08-20).
R8  a debt stated to the debtor: "you owe me 20" / "sure" fires; "alright" / "yeah i know" /
    "i remember" is only admitting it and stays quiet (Akash 09-17, ruling 8).

Narrow on purpose, like rulings_0915_relabel.py: a window is touched only when the pattern
is unambiguous, and samples print so every rule is read before it is applied.

    python data_gen/rulings_0917_relabel.py conv_windows_v16r.zip             # report only
    python data_gen/rulings_0917_relabel.py conv_windows_v16r.zip --apply OUT  # write
"""
import argparse, io, json, random, re, sys, zipfile
from collections import Counter
from pathlib import Path

# stdout is wrapped as UTF-8 by rulings_0915_relabel on import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_gen"))
from rulings_0915_relabel import REQUEST_SHAPE, CANCEL, OWN_CAR, NEED_ONLY, MONEY_ACT, RIDE_KW, MONEY_STRICT  # noqa: E402

BARE = re.compile(r"^\W*(ok|okay|k|kk|okie|sure|cool|nice|alright|aight|great|perfect|yes|yeah|yep|yup)\W*$", re.I)
CLEAR_TAKE = re.compile(
    r"^\W*((ok|okay|sure|yeah|yes)[\s,!.]+)?"
    r"(let me (book|send|pay|get|order|transfer|cover)\b"
    r"|i'?ll (book|send|pay|transfer|get|order|cover) (it|one|that|the|you|a|an|for)\b"
    r"|i will (book|send|pay|transfer) "
    r"|(on it[,\s]+)?(sending|booking|ordering|paying|transferring)( it| that| the \w+)? (now|rn|right now)\b)", re.I)
NOT_TAKE = re.compile(r"\?|\b(later|tomorrow|tonight|friday|next week|after|once|when|if|maybe|can'?t|not|no)\b", re.I)
LET_ME_ANSWER = re.compile(r"^\W*((ok|okay|sure|yeah|yes|alright)[\s,!.]+)?let me (book|send|pay|get|order|transfer|cover)\b", re.I)
DEBT = re.compile(r"\b(you|u) (still )?owe me\b|\b(you|u) (still )?(haven'?t|havent|never) paid me\b", re.I)
PAY_AGREE = re.compile(r"^\W*(sure|sure thing|sure np|yeah let me pay|let me pay)\W*$"
                       r"|\b(sending|paying|transferring)( it| you| that)? (now|rn|right now)\b", re.I)
ADMIT = re.compile(r"^\W*(alright|yeah i know|i know|i remember|i know i know|yes i know|noted|true|right"
                   r"|i haven'?t forgotten|yeah yeah( i know)?)\W*$", re.I)


def prev_other(win, i, who):
    """Index of the latest message before i not written by `who`."""
    j = i - 1
    while j >= 0 and win[j]["s"] == who:
        j -= 1
    return j


def intent_of(text):
    booking = bool(re.search(r"\b(book|order|call|grab|get)\b", text, re.I))
    if MONEY_ACT.search(text) and not booking:
        return "money"
    if RIDE_KW.search(text) or re.search(r"\bbook\b", text, re.I):
        return "ride"
    if MONEY_STRICT.search(text):
        return "money"
    return None


def r6(w):
    win, last = w["window"], w["window"][-1]
    if not w["fire"] or not BARE.search(last["t"]) or len({m["s"] for m in win}) < 3:
        return None
    c = last["s"]
    b_i = len(win) - 2
    if b_i < 1 or win[b_i]["s"] == c:
        return None
    b = win[b_i]
    if not CLEAR_TAKE.search(b["t"]) or NOT_TAKE.search(b["t"]):
        return None
    a_i = prev_other(win, b_i, b["s"])
    if a_i < 0 or win[a_i]["s"] == c or not REQUEST_SHAPE.search(win[a_i]["t"]) or CANCEL.search(win[a_i]["t"]):
        return None
    return []


def r7(w):
    win, last = w["window"], w["window"][-1]
    if w["fire"] or not LET_ME_ANSWER.search(last["t"]) or NOT_TAKE.search(last["t"]) or OWN_CAR.search(last["t"]):
        return None
    a_i = prev_other(win, len(win) - 1, last["s"])
    if a_i != len(win) - 2:                  # the answer must follow the request directly
        return None
    req = win[a_i]["t"]
    if not REQUEST_SHAPE.search(req) or CANCEL.search(req) or OWN_CAR.search(req) or NEED_ONLY.search(req):
        return None
    want = intent_of(req)
    verb_ride = bool(re.search(r"let me (book|get|order)", last["t"], re.I))
    if want is None or (want == "ride") != verb_ride:
        return None                          # the answer's verb must match the request
    return [want]


def r8(w):
    win, last = w["window"], w["window"][-1]
    a_i = prev_other(win, len(win) - 1, last["s"])
    if a_i != len(win) - 2:
        return None
    stmt = win[a_i]["t"]
    if not DEBT.search(stmt) or REQUEST_SHAPE.search(stmt) or CANCEL.search(stmt):
        return None
    if not w["fire"] and PAY_AGREE.search(last["t"]) and not NOT_TAKE.search(last["t"]):
        return ["money"]
    if w["fire"] and ADMIT.search(last["t"]):
        return []
    return None


RULES = {"R6 clear take, 3rd person bare ok -> quiet": r6,
         "R7 'let me X' answering a request -> fires": r7,
         "R8 debt: 'sure' fires / admit quiet": r8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--apply", default=None)
    ap.add_argument("--samples", type=int, default=12)
    a = ap.parse_args()
    z = zipfile.ZipFile(ROOT / a.src)
    rnd = random.Random(17)
    out = {}
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
            print(f"    {name:44} {len(h):6,}   roles {Counter(w.get('role') for w, _ in h).most_common(4)}")
            if split == "train.json":
                for w, new in rnd.sample(h, min(a.samples, len(h))):
                    print(f"        {w['fire']} -> {new}   " + "  |  ".join(f"{m['s']}: {m['t'][:50]}" for m in w["window"][-4:]))
        if a.apply:
            for name, h in hits.items():
                for w, new in h:
                    w["fire"] = new
                    w["relabel_0917"] = name.split()[0]
        out[split] = rows
    if a.apply:
        with zipfile.ZipFile(ROOT / a.apply, "w", zipfile.ZIP_DEFLATED) as zo:
            for split, rows in out.items():
                zo.writestr(split, json.dumps(rows, ensure_ascii=False))
        print(f"\n  wrote {a.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
