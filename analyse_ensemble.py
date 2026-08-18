"""Are v5, v8 and v9 making the SAME mistakes, and would combining them help?

Reads the per-turn recordings from record_decisions.py and answers three questions
without touching a server:

  1. How much do the models' errors overlap? If they fail on the same turns, no
     combining rule can help and the differences between them are cosmetic.
  2. Would a vote beat the best single model?
  3. Would an asymmetric rule beat it - fire when ANY model fires (max recall) or only
     when ALL agree (max precision)?

Scored the same way as the human eval, so the numbers are comparable: a fire
conversation is correct if the right intent fires anywhere in it; a quiet conversation
is correct only if nothing fires at all.

    python analyse_ensemble.py
"""
import json, sys, io
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
MODELS = ["v5", "v8", "v9"]


def load():
    d = {}
    for m in MODELS:
        p = ROOT / "data/eval" / f"DEC_{m}.json"
        if not p.exists():
            print(f"  missing {p.name} — run record_decisions.py for {m}")
            return None
        d[m] = {(r["conv"], r["turn"]): r for r in json.loads(p.read_text(encoding="utf-8"))}
    return d


def conv_meta(dec):
    meta = {}
    for k, r in dec[MODELS[0]].items():
        meta[r["conv"]] = (r["expects_fire"], r["scenario"], r["kind"], r["tier"])
    return meta


def score(fire_fn, keys, dec, meta):
    """fire_fn(key) -> list of intents. Returns (fire_ok, fire_n, quiet_ok, quiet_n)."""
    per_conv = defaultdict(list)
    for k in keys:
        per_conv[k[0]].append(k)
    fo = fn = qo = qn = 0
    for conv, ks in per_conv.items():
        expects, *_ = meta[conv]
        want = None
        for k in ks:
            w = dec[MODELS[0]][k]["want"]
            if w:
                want = w[0]
        fired_any = False
        fired_right = False
        for k in sorted(ks, key=lambda x: x[1]):
            got = fire_fn(k)
            if got:
                fired_any = True
                if want and want in got:
                    fired_right = True
        if expects:
            fn += 1
            fo += fired_right
        else:
            qn += 1
            qo += not fired_any
    return fo, fn, qo, qn


def report(name, fire_fn, keys, dec, meta):
    fo, fn, qo, qn = score(fire_fn, keys, dec, meta)
    tot = (fo + qo) / max(fn + qn, 1)
    print(f"  {name:28} fires {fo/max(fn,1):6.1%}   quiet {qo/max(qn,1):6.1%}   "
          f"overall {tot:6.1%}")
    return tot


def main():
    dec = load()
    if dec is None:
        return 1
    keys = sorted(set(dec[MODELS[0]]) & set(dec[MODELS[1]]) & set(dec[MODELS[2]]))
    meta = conv_meta(dec)
    print(f"  {len(keys)} turns recorded for all three models\n")

    # --- 1. error overlap -------------------------------------------------------
    wrong = {}
    for m in MODELS:
        wrong[m] = {k for k in keys
                    if bool(dec[m][k]["got"]) != bool(dec[m][k]["want"])
                    or (dec[m][k]["want"] and dec[m][k]["want"][0] not in dec[m][k]["got"])}
    print("  per-turn errors")
    for m in MODELS:
        print(f"    {m}  {len(wrong[m])}")
    print("\n  overlap")
    for a, b in combinations(MODELS, 2):
        inter = len(wrong[a] & wrong[b])
        union = len(wrong[a] | wrong[b])
        print(f"    {a}&{b}  shared {inter:>5}   jaccard {inter/max(union,1):.2f}")
    allthree = wrong[MODELS[0]] & wrong[MODELS[1]] & wrong[MODELS[2]]
    anyone = wrong[MODELS[0]] | wrong[MODELS[1]] | wrong[MODELS[2]]
    print(f"    all three wrong on {len(allthree)} turns")
    print(f"    at least one wrong on {len(anyone)}")
    print(f"    -> {len(allthree)/max(len(anyone),1):.0%} of all errors are shared by "
          f"every model (irreducible by combining)")

    # --- 2. single models, for reference ----------------------------------------
    print("\n  single models")
    best = 0
    for m in MODELS:
        best = max(best, report(m, lambda k, m=m: dec[m][k]["got"], keys, dec, meta))

    # --- 3. combining rules -----------------------------------------------------
    def vote(k):
        c = defaultdict(int)
        for m in MODELS:
            for i in dec[m][k]["got"]:
                c[i] += 1
        return [i for i, n in c.items() if n >= 2]

    def any_fire(k):
        out = []
        for m in MODELS:
            for i in dec[m][k]["got"]:
                if i not in out:
                    out.append(i)
        return out

    def all_fire(k):
        sets = [set(dec[m][k]["got"]) for m in MODELS]
        return sorted(set.intersection(*sets))

    print("\n  combining rules")
    r1 = report("majority (2 of 3)", vote, keys, dec, meta)
    r2 = report("any model fires", any_fire, keys, dec, meta)
    r3 = report("all three agree", all_fire, keys, dec, meta)

    print(f"\n  best single {best:.1%}   best combined {max(r1, r2, r3):.1%}")
    if max(r1, r2, r3) <= best:
        print("  -> combining does NOT beat the best single model; the errors are shared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
