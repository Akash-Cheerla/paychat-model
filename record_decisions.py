"""Replay human_eval through one model and record every per-turn decision to disk.

Scoring a model tells you its number. Recording its DECISIONS lets you ask questions
afterwards without paying for another 31,432 requests each time — how correlated two
models' mistakes are, whether a vote would beat any single one, what a different
threshold would have done.

That matters here because v5, v8 and v9 have opposite strengths on the human eval:

    v5   fires 81.7%   quiet 74.6%
    v8   fires 74.8%   quiet 82.4%
    v9   fires 80.0%   quiet 78.6%

If their errors land in different places, combining them beats all three at zero
training cost. If the errors are the same errors, no combining rule can help and that is
worth knowing before building one.

One model per run, sequentially — three servers at once got OOM-killed mid-run before,
and the scorer swallows per-request errors, so a dead server quietly reports "never
fired" for every turn instead of failing.

    python record_decisions.py --url http://127.0.0.1:8030/detect --label v5
"""
import argparse, json, sys, io, time
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
MR = ("money", "ride")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    convs = json.loads((ROOT / "data/eval/human_eval.json").read_text(encoding="utf-8"))
    if a.limit:
        convs = convs[:a.limit]

    tag = f"{a.label}_{int(time.time())}"
    rows, dead = [], 0
    for ci, c in enumerate(convs):
        room = f"rec_{tag}_{ci}"
        for ti, t in enumerate(c["turns"]):
            try:
                d = requests.post(a.url, timeout=60, json={
                    "text": t["text"], "room_id": room,
                    "sender": str(t.get("sender", "A")),
                    "message_id": f"{tag}_{ci}_{ti}"}).json()
            except Exception:
                dead += 1
                continue
            cs = (d.get("conversation_state") or {}).get("scores") or {}
            rows.append({
                "conv": ci, "turn": ti,
                "scenario": c["scenario"], "kind": c["kind"], "tier": c["tier"],
                "expects_fire": c["expects_fire"],
                "want": t.get("fire") or [],
                "got": [i for i in (d.get("intents") or []) if i in MR],
                "conv_money": round(cs.get("money", 0.0), 4),
                "conv_ride": round(cs.get("ride", 0.0), 4),
            })
        if (ci + 1) % 200 == 0:
            print(f"  {ci+1}/{len(convs)} conversations", flush=True)

    # A run with failed requests is not a smaller run, it is a WRONG one: a missing
    # response is indistinguishable from "the model stayed silent". Refuse to save it.
    if dead:
        print(f"\n  {dead} requests failed — refusing to write a corrupted recording")
        return 1

    out = ROOT / "data/eval" / f"DEC_{a.label}.json"
    out.write_text(json.dumps(rows), encoding="utf-8")
    fired = sum(1 for r in rows if r["got"])
    print(f"\n  wrote {out.name}: {len(rows)} turns, {fired} fired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
