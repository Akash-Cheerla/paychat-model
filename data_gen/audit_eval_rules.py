"""Find eval labels that contradict FIRING_RULE.md.

An eval that disagrees with the spec cannot decide a ship. We rejected v7 partly on
`mixed_eval`, and that set expects a ride prompt for "someone needs to drive me, i'm
carless" / "ill swing by at 1130" — a friend's own car, which §4 says is never a ride.
The model was right and the eval was wrong, and it counted against the model.

Every check below cites the rule it enforces. A check that cannot cite one does not
belong here — that is the mistake this script exists to correct, and it is easy to
repeat while writing it.

  python data_gen/audit_eval_rules.py                    # report only
  python data_gen/audit_eval_rules.py --fix              # rewrite the labels
  python data_gen/audit_eval_rules.py --only mixed_eval
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ["mixed_eval.json", "group_eval_v6.json", "claude_eval.json",
         "behaviour_eval.json", "ship_eval.json", "group_heldout.json"]

# §4 — ride means ride-HAILING. A friend driving you is not a booking.
HAILING = re.compile(
    r"\b(cab|taxi|uber|ola|lyft|rapido|rickshaw|"
    r"ride\s*share|bolt|didi)\b", re.IGNORECASE)
# Someone asking a PERSON to drive them. Their own car is never a booking (§4).
# Matched across the WHOLE conversation, not just the acking turn: the turn that fires
# is normally a bare "sure" or "ya ill be there by 6:45", which carries no evidence
# either way. The evidence is in what it answers. Checking the wrong turn is why this
# check missed ~60 labels.
# Procuring a vehicle FOR someone, without ever naming it. Found by hand-checking the
# first eight flags: "can u grab me a ride there?" / "lol sure. sending one to ur
# building now" was flagged as a friend's own car, but "sending one" is a booking. One
# bad flag in eight would have corrupted ~8 of 69 labels. Anything matching this is a
# real ride however the conversation is worded, so it is checked first and wins.
PROCURES = re.compile(
    r"\b(sending\s+(?:one|it)\b|book(?:ing|ed)?\s+(?:one|it|u|you)\b|"
    r"order(?:ing|ed)?\s+(?:one|it)\b|call(?:ing)?\s+(?:one|a\s+car)\b|"
    r"got\s+(?:u|you)\s+one\b|sent\s+one\b|on\s+its\s+way\s+to\s+(?:u|you)\b)",
    re.IGNORECASE)
OWN_CAR_REQ = re.compile(
    r"\b(pick\s+(?:me|us|him|her)\s+up|drop\s+(?:me|us)\b|grab\s+(?:me|us)\b|"
    r"scoop\s+(?:me|us)\b|give\s+(?:me|us)\s+a\s+(?:lift|ride)|"
    r"come\s+(?:and\s+)?get\s+(?:me|us)|can\s+(?:u|you)\s+come|take\s+me\b|"
    r"need\s+a\s+(?:lift|ride)|any(?:one|body)\s+free\s+to\s+(?:grab|get|drop))\b",
    re.IGNORECASE)
OWN_CAR = re.compile(
    r"\b(i'?ll\s+(?:drive|drop|pick)|pick\s+(?:you|u)\s+up|drop\s+(?:you|u)\b|"
    r"swing\s+by|come\s+get\s+(?:you|u)|give\s+(?:you|u)\s+a\s+lift|"
    r"in\s+my\s+car|i'?m\s+driving)\b", re.IGNORECASE)

# §3a — a completed action fires nothing, however it is phrased.
COMPLETED = re.compile(
    r"\b(already\s+(?:sent|paid|booked|did|done)|just\s+(?:sent|paid|booked)|"
    r"sent\s+it\s+(?:already|yesterday|earlier|this\s+morning|last\s+week)|"
    r"transferred\s+(?:it\s+)?(?:already|earlier)|booked\s+(?:it\s+)?already|"
    r"driver\s+is\s+(?:here|on\s+the\s+way)|cab\s+is\s+(?:here|outside))\b",
    re.IGNORECASE)

# A commitment still pointing FORWARD. When this and COMPLETED both match, the message
# says two contradictory things — "i got you, i'll book it, just sent the confirmation"
# — and a person has to decide. Reported, never rewritten.
FORWARD = re.compile(
    r"\b(i'?ll\s+(?:send|pay|book|transfer|get|do)|sending\s+now|"
    r"booking\s+(?:it\s+)?now|on\s+it|doing\s+it\s+now|will\s+send|will\s+book)\b",
    re.IGNORECASE)


def turns_of(c):
    return c.get("turns") or c.get("messages") or []


def label_of(t):
    v = t.get("fire", t.get("expected"))
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def set_label(t, val):
    t["fire" if "fire" in t else "expected"] = val


def check(conv):
    """Return [(turn_index, reason, current, corrected)] for this conversation."""
    ts = turns_of(conv)
    out = []
    whole = " ".join((t.get("text") or "") for t in ts)
    mentions_hailing = bool(HAILING.search(whole))
    asks_for_a_driver = bool(OWN_CAR_REQ.search(whole) or OWN_CAR.search(whole))

    for i, t in enumerate(ts):
        lab = label_of(t)
        txt = t.get("text") or ""

        # §4 — a ride label where no hired vehicle is mentioned ANYWHERE in the
        # conversation and what was asked for was a person driving. Confirmed with
        # the product owner 2026-08-14: no booking intent, no prompt.
        if ("ride" in lab and not mentions_hailing and asks_for_a_driver
                and not PROCURES.search(txt)):
            out.append((i, "§4 friend's own car, no booking anywhere", lab,
                        [x for x in lab if x != "ride"]))
            continue

        # §3a — the action is reported as finished, so there is nothing left to do.
        # A message carrying BOTH a forward commitment and a completion marker
        # ("i got you, i'll book it, just sent the confirmation") is genuinely
        # ambiguous. Report it, never rewrite it — silently flipping a label we are
        # not sure about is how an eval stops describing the rule.
        if lab and COMPLETED.search(txt):
            if FORWARD.search(txt):
                out.append((i, "AMBIGUOUS: commitment + completion (review by hand)",
                            lab, None))
            else:
                out.append((i, "§3a completed action", lab, []))
            continue

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    grand = Counter()
    for name in EVALS:
        if a.only and a.only not in name:
            continue
        p = ROOT / "data/eval" / name
        if not p.exists():
            print(f"  {name}: missing")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = d if isinstance(d, list) else d.get("conversations", [])

        hits, shown = [], 0
        for ci, c in enumerate(rows):
            for ti, why, cur, fix in check(c):
                hits.append((ci, ti, why, cur, fix))

        print(f"\n{'='*66}\n{name}   {len(rows)} conversations   "
              f"{len(hits)} label(s) contradict the rule\n{'='*66}")
        by = Counter(h[2] for h in hits)
        for why, n in by.most_common():
            print(f"  {n:>4}  {why}")
            grand[why] += n
        for ci, ti, why, cur, fix in hits[:a.show]:
            ts = turns_of(rows[ci])
            print(f"\n   conv {ci} turn {ti}:  {cur} -> {fix}   ({why})")
            for t in ts[max(0, ti-2):ti+1]:
                print(f"      {t.get('sender')}: {(t.get('text') or '')[:64]}")

        if a.fix and hits:
            # Snapshot BEFORE mutating. `rows` is `d` when the file is a bare list, and
            # set_label edits the turn dicts in place, so writing the backup after the
            # loop wrote the already-corrected labels and silently destroyed the
            # original. mixed_eval.pre_rule_audit.json was produced that way and is not
            # the pre-audit state despite its name.
            bak = p.with_suffix(".pre_rule_audit.json")
            if not bak.exists():
                bak.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            applied = 0
            for ci, ti, why, cur, fix in hits:
                if fix is None:          # ambiguous — reported, never rewritten
                    continue
                set_label(turns_of(rows[ci])[ti], fix)
                applied += 1
            p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            print(f"\n  FIXED {applied} labels "
                  f"({len(hits)-applied} ambiguous, left for review); "
                  f"original kept at {bak.name}")

    print(f"\n{'='*66}\ntotal across all sets: {sum(grand.values())}")
    for why, n in grand.most_common():
        print(f"  {n:>4}  {why}")
    if not a.fix:
        print("\n(report only — pass --fix to rewrite)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
