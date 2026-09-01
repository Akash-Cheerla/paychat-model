"""Replay whole rooms from a dogfood log and compare what fires now with what fired then.

The log records what the deployed model did, message by message, including the
conversation score. So the same rooms replayed through a changed build give a direct
before/after on real traffic - no reconstruction, no invented labels.

The number that matters is messages the classifier scored high on that produced no
prompt. Those are the pipeline overriding a confident model, and on 2026-08-30 there
were sixteen of them.

    python replay_log_rooms.py --log data/eval/dogfood_2026-08-30.jsonl \\
                               --url http://127.0.0.1:8900/detect
"""
import argparse, json, sys, io, time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.stdout = io.TextIOWrapper(open(sys.__stdout__.fileno(), "wb", closefd=False),
                              encoding="utf-8", errors="replace", line_buffering=True)
MR = ("money", "ride")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--min-score", type=float, default=0.90)
    a = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / a.log).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["room"], []).append(r)

    tag = int(time.time())
    was_sup, now_sup, recovered, newly_sup = 0, 0, [], []
    fires_then = fires_now = 0

    for room, rs in by.items():
        for i, r in enumerate(rs):
            try:
                d = requests.post(a.url, timeout=60, json={
                    "text": r["text"], "room_id": f"rl_{tag}_{room}",
                    "sender": str(r["sender"]), "message_id": f"{tag}_{room}_{i}"}).json()
            except Exception:
                continue
            now = [x for x in (d.get("intents") or []) if x in MR]
            then = [x for x in (r.get("fired") or []) if x in MR]
            fires_then += bool(then)
            fires_now += bool(now)

            # the log's own score for this message, as the deployed model saw it
            c = r.get("conv") or {}
            for intent in MR:
                s = float(c.get(intent, 0) or 0)
                if s < a.min_score:
                    continue
                t_sup = intent not in then
                n_sup = intent not in now
                was_sup += t_sup
                now_sup += n_sup
                if t_sup and not n_sup:
                    recovered.append((room, intent, s, rs[i-1]["text"][:34] if i else "",
                                      r["sender"], r["text"][:52]))
                if n_sup and not t_sup:
                    newly_sup.append((room, intent, s, r["sender"], r["text"][:52]))

    print(f"\n  {sum(len(v) for v in by.values())} messages, {len(by)} rooms")
    print(f"  prompts   then {fires_then}   now {fires_now}")
    print(f"\n  scored >= {a.min_score} but no prompt:")
    print(f"    then {was_sup}")
    print(f"    now  {now_sup}")
    print(f"\n  RECOVERED (was silent, now fires): {len(recovered)}")
    for room, intent, s, prev, snd, txt in recovered[:20]:
        print(f"    [{room}] {intent} {s:.3f}  after {prev!r}")
        print(f"        {snd}: {txt!r}")
    print(f"\n  NEWLY SUPPRESSED (fired then, silent now): {len(newly_sup)}")
    for room, intent, s, snd, txt in newly_sup[:20]:
        print(f"    [{room}] {intent} {s:.3f}  {snd}: {txt!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
