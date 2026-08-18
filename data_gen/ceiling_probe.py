"""Sample the decision points a second annotator has to rule on, labels hidden.

The point: before spending another week chasing 90%, find out whether 90% is reachable.
If two careful people applying FIRING_RULE.md to the same conversations agree only 88%
of the time, then 88% is the ceiling and every point we are chasing above it is noise
we are fitting. Nobody has measured this.

Sampling is deliberately NOT uniform. Most turns in any eval are obvious neutrals
("lol", "ok cool") that both annotators get right, and including them inflates
agreement toward 99% while telling us nothing. We sample the turns where a decision is
actually made:

  * turns the stored label says fire
  * turns the model fires but the label does not, or vice versa
  * turns adjacent to a fire, where the commit/prepare/complete line falls

Emits a review file with the stored label REMOVED, so the second pass cannot anchor on
it. Adjudicate, then run --score to compare.

  python data_gen/ceiling_probe.py --n 50 --out ceiling_batch1.json
  python data_gen/ceiling_probe.py --score ceiling_batch1.json
"""
import argparse, json, random, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
EVALS = ["behaviour_eval.json", "ship_eval.json", "mixed_eval.json", "group_heldout.json"]


def labels(t):
    v = t.get("fire", t.get("expected"))
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def decision_points(conv, src, ci):
    ts = conv.get("turns") or conv.get("messages") or []
    fire_idx = {i for i, t in enumerate(ts) if labels(t)}
    # A fire, and the turn either side of it — the boundary is where annotators split.
    keep = set()
    for i in fire_idx:
        keep |= {i - 1, i, i + 1}
    out = []
    for i in sorted(x for x in keep if 0 <= x < len(ts)):
        out.append({
            "src": src, "conv": ci, "turn": i,
            "kind": conv.get("kind", "DM"), "scenario": conv.get("scenario", ""),
            "context": [{"sender": t.get("sender"), "text": t.get("text")}
                        for t in ts[max(0, i - 5):i + 1]],
            "stored": labels(ts[i]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="ceiling_batch1.json")
    ap.add_argument("--score")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    if a.score:
        d = json.loads((ROOT / "data/eval" / a.score).read_text(encoding="utf-8"))
        items = [i for i in d["items"] if i.get("mine") is not None]
        if not items:
            print("  no adjudicated items yet — fill in \"mine\" on each"); return 1
        agree = sum(1 for i in items
                    if sorted(i["mine"]) == sorted(i["stored_hidden"]))
        print(f"  adjudicated {len(items)} decision points")
        print(f"  agreement   {agree}/{len(items)} = {agree/len(items):.1%}")
        print("\n  --- disagreements ---")
        for i in items:
            if sorted(i["mine"]) != sorted(i["stored_hidden"]):
                print(f"    {i['src']}#{i['conv']}t{i['turn']}  "
                      f"stored={i['stored_hidden']} mine={i['mine']}")
                print(f"      {i['context'][-1]['text'][:70]}")
        return 0

    pool = []
    for name in EVALS:
        p = ROOT / "data/eval" / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = d if isinstance(d, list) else d.get("conversations", [])
        for ci, conv in enumerate(rows):
            pool.extend(decision_points(conv, name.replace(".json", ""), ci))

    random.seed(a.seed)
    random.shuffle(pool)
    picked = pool[:a.n]
    out = {"note": "fill in \"mine\" for each item without reading stored_hidden",
           "items": [{**{k: v for k, v in it.items() if k != "stored"},
                      "stored_hidden": it["stored"], "mine": None} for it in picked]}
    dst = ROOT / "data/eval" / a.out
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  pool {len(pool)} decision points -> sampled {len(picked)}")
    print(f"  wrote data/eval/{a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
