"""One command: does a candidate model beat v14 on everything that matters?

Runs the robustness suite and replays the real dogfood logs through the candidate, scores
both the way this project scores them (timing-aware, a prompt up to 2 messages off still
counts), and prints PASS/FAIL against v14's measured numbers.

v14's baselines were measured on 2026-09-15 against the adjudications revised to the
2026-09-13/15 rulings. Re-measure them (--baseline) if the labels change again.

    python tests/ship_gate.py --url http://127.0.0.1:8971/detect --name v17
    python tests/ship_gate.py --url http://127.0.0.1:8972/detect --name v14 --baseline

Batteries and the 82-case set are separate commands and still count:
    python tests/run_all.py --url v17=<url> --out <dir>
    python data_gen/llm_context_group_test.py --url <url> --ours-only --save <json>
"""
import argparse, io, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from replay_ab import load, play      # noqa: E402  (wraps stdout as UTF-8)

MR = ("money", "ride")
WINDOW = 2
LOGS = ("dogfood_2026-09-10", "dogfood_2026-08-11")
# PRODUCTION as committed: v14 on git HEAD's app.py (7b172f0), measured 2026-09-17 on the
# 845-case suite (exact label set) and the rulings-revised labels. v14 on the uncommitted app.py
# scored 72.0/12.3 and 82.5/15.6 on the logs; the gate compares against what users have today.
BASE = {
    "suite_fire": 64.2, "suite_quiet": 95.3, "suite_heldout": 71.4,
    "dogfood_2026-09-10": {"caught": 72.0, "wrong": 13.5},
    "dogfood_2026-08-11": {"caught": 82.5, "wrong": 16.5},
}


def score_log(stem, url, tag):
    rooms = load(str(ROOT / f"data/eval/{stem}.jsonl"))
    adj = {k: v for k, v in json.loads((ROOT / f"data/eval/ADJUDICATED_{stem}.json").read_text(encoding="utf-8")).items()
           if not k.startswith("_")}
    fires = {}
    for k, (rid, msgs) in enumerate(rooms.items()):
        for i, _, _, g in play(url, rid, msgs, f"{tag}{k}"):
            fires[f"{rid}#{i}"] = next((x for x in g if x in MR), None)
    moments = [(k, v) for k, v in adj.items() if v in MR]
    other = {k for k, _ in moments}
    caught, used = 0, set()
    for k, want in moments:
        rid, i = k.rsplit("#", 1)
        i = int(i)
        if fires.get(k) == want:
            caught += 1
            used.add(k)
            continue
        hit = next((f"{rid}#{j}" for j in range(i - WINDOW, i + WINDOW + 1)
                    if j != i and f"{rid}#{j}" not in used and fires.get(f"{rid}#{j}") == want
                    and f"{rid}#{j}" not in other), None)
        if hit:
            caught += 1
            used.add(hit)
    wrong = sum(1 for k, v in adj.items() if v == "none" and fires.get(k) and k not in used)
    shown = sum(1 for v in fires.values() if v)
    return (100 * caught / len(moments), 100 * wrong / max(shown, 1),
            f"{caught}/{len(moments)} caught, {wrong}/{shown} prompts wrong")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--baseline", action="store_true", help="print the numbers as a new baseline")
    a = ap.parse_args()

    out = subprocess.run([sys.executable, str(ROOT / "tests/robustness_suite.py"),
                          "--url", a.url, "--name", a.name],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    got = {}
    for line in out.splitlines():
        for key, label in (("suite_fire", "must fire, fired:"), ("suite_quiet", "must stay quiet, quiet:"),
                           ("suite_heldout", "held-out phrasings:")):
            if label in line:
                got[key] = float(line.rsplit("=", 1)[1].strip().rstrip("%"))
    rows = [(f"robustness: must fire", got.get("suite_fire", 0), BASE["suite_fire"], ">"),
            (f"robustness: must stay quiet", got.get("suite_quiet", 0), BASE["suite_quiet"], ">="),
            (f"robustness: held-out phrasings", got.get("suite_heldout", 0), BASE["suite_heldout"], ">")]
    # Real traffic: no worse than v14 on either number in either log, and strictly better on at
    # least one. "Strictly more catches" alone would fail a model that only cuts wrong prompts.
    better = False
    for stem in LOGS:
        c, w, detail = score_log(stem, a.url, f"gate{a.name}")
        short = stem.replace("dogfood_", "")
        rows.append((f"{short}: commitments caught", c, BASE[stem]["caught"], ">="))
        rows.append((f"{short}: wrong prompts", w, BASE[stem]["wrong"], "<="))
        better |= c > BASE[stem]["caught"] + 0.05 or w < BASE[stem]["wrong"] - 0.05
        print(f"  {short}: {detail}")

    print(f"\n  {'criterion':38}{a.name:>10}{'v14':>10}   verdict")
    ok = True
    for name, val, base, op in rows:
        good = {">": val > base, ">=": val >= base - 0.05, "<=": val <= base + 0.05}[op]
        ok &= good
        print(f"  {name:38}{val:>9.1f}%{base:>9.1f}%   {'PASS' if good else 'FAIL'}  (needs {op} v14)")
    print(f"  {'real traffic strictly better somewhere':38}{'':>20}   {'PASS' if better else 'FAIL'}")
    ok &= better
    print(f"\n  {'SHIP' if ok else 'DO NOT SHIP'} - batteries and the 82-case set are separate, see the docstring")
    if a.baseline:
        print("\n  BASE = " + json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in got.items()}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
