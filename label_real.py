"""Label REAL conversations with the teacher, and show where the live model disagreed.

This is the piece that has been missing since the start. Every label our models have
trained on came from DeepSeek writing short synthetic conversations - median 6 turns,
against 15-40 in real chats. That mismatch is why six training rounds went nowhere.

Input is a dogfood log: real team traffic, one JSON object per message, already carrying
what the model actually did (`fired`). So each turn gets two opinions - the teacher's and
the live model's - and the disagreements are the only thing a human has to read.

The output stays gitignored. It is derived from real private messages and carries the
same restriction as its source: measurement only.

    python label_real.py --log data/eval/dogfood_2026-08-21.jsonl \\
                         --out data/eval/REAL_dogfood_0821.json
"""
import argparse, json, sys, io, time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_teacher import SYSTEM, SHOTS, API, window          # one prompt, one place

# stdout is already wrapped for utf-8 by llm_teacher at import; wrapping it a second
# time here closed the underlying buffer as soon as the first wrapper was collected.
ROOT = Path(__file__).resolve().parent


def ask(key, model, text, retries=4):
    msgs = [{"role": "system", "content": SYSTEM}]
    for w, a in SHOTS:
        msgs += [{"role": "user", "content": w}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": text})
    for i in range(retries):
        try:
            r = requests.post(API, timeout=90, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "temperature": 0, "max_tokens": 4, "messages": msgs})
            if r.status_code == 200:
                w = r.json()["choices"][0]["message"]["content"].strip().lower()
                return "money" if "money" in w else ("ride" if "ride" in w else "none")
            if r.status_code in (401, 402):
                return f"ERR{r.status_code}"
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return "ERR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--min-turns", type=int, default=4)
    a = ap.parse_args()

    key = ""
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK_API_KEY"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
    assert key, "no DEEPSEEK_API_KEY"

    rows = [json.loads(l) for l in (ROOT / a.log).read_text(encoding="utf-8").splitlines() if l.strip()]
    rooms = {}
    for r in rows:
        rooms.setdefault(r["room"], []).append(r)
    rooms = {k: v for k, v in rooms.items() if len(v) >= a.min_turns}
    turns_total = sum(len(v) for v in rooms.values())
    print(f"  {len(rooms)} rooms, {turns_total} messages to label\n")

    jobs = [(room, k) for room, rs in rooms.items() for k in range(len(rs))]

    def one(job):
        room, k = job
        rs = rooms[room]
        w = window([{"sender": r["sender"], "text": r["text"]} for r in rs], k)
        return room, k, ask(key, a.model, w)

    labels = {}
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for room, k, v in ex.map(one, jobs):
            if v.startswith("ERR"):
                print(f"  API error {v} - stopping"); return 1
            labels[(room, k)] = v
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)}")

    out, agree, dis = [], 0, []
    for room, rs in rooms.items():
        turns = []
        for k, r in enumerate(rs):
            teach = labels[(room, k)]
            live = [i for i in (r.get("fired") or []) if i in ("money", "ride")]
            live1 = live[0] if live else "none"
            if teach == live1:
                agree += 1
            else:
                dis.append((room, k, r["sender"], r["text"][:60], live1, teach))
            turns.append({"sender": r["sender"], "text": r["text"],
                          "fire": [teach] if teach != "none" else [],
                          "live_model": live, "ts": r.get("ts")})
        out.append({"room": room, "source": Path(a.log).name,
                    "expects_fire": any(t["fire"] for t in turns), "turns": turns})

    (ROOT / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    n = len(jobs)
    print(f"\n  labelled {n} messages across {len(out)} conversations")
    print(f"  teacher and live model agree   {agree}/{n} = {agree/n*100:.1f}%")
    print(f"  disagree                       {len(dis)}\n")
    kinds = Counter((d[4], d[5]) for d in dis)
    for (live1, teach), c in kinds.most_common():
        print(f"    live={live1:6} teacher={teach:6}  {c}")
    print("\n  first disagreements to review:\n")
    for room, k, snd, txt, live1, teach in dis[:20]:
        print(f"    {room:12} {snd}: {txt!r}")
        print(f"                 live={live1}  teacher={teach}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
