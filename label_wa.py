"""Label real WhatsApp conversations with the teacher.

The three Windsor ride groups are the most valuable negatives we have. They are
ride-SHARING groups - people offering seats in their own cars - and FIRING_RULE 4 says
that is not ride-hailing and must never fire. So they read exactly like the intent we do
fire on, and every one of them should stay silent. Nothing in our synthetic data covers
that, and no amount of DeepSeek prompting would have invented it.

Messages are chunked into fixed conversations rather than split on time gaps: the
classifier reads a 10-message window regardless, so the chunking only decides where the
window starts, and fixed chunks keep the labelling reproducible.

Output is gitignored - these are third-party private messages, measurement only, and no
verbatim text may go into training data.

    python label_wa.py --chat "WhatsApp Chat - RIDES WINDSOR" --cap 600 \\
                       --out data/eval/REAL_wa_rides1.json
"""
import argparse, io, json, sys, time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_gen"))
from llm_teacher import SYSTEM, SHOTS, API, window
import wa_study

# Both llm_teacher and wa_study replace sys.stdout with their own utf-8 wrapper at
# import. The second replacement drops the first, and when the first is collected it
# closes the shared buffer - every later print raises 'I/O operation on closed file'.
# Reopen a fresh stream on the original descriptor instead, with closefd False so
# nothing else can close it out from under us.
sys.stdout = io.TextIOWrapper(open(sys.__stdout__.fileno(), 'wb', closefd=False),
                              encoding='utf-8', errors='replace', line_buffering=True)


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
    ap.add_argument("--chat", required=True, help="folder name under Downloads")
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    key = ""
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK_API_KEY"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
    assert key, "no DEEPSEEK_API_KEY"

    p = Path(f"C:/Users/akash/Downloads/{a.chat}/_chat.txt")
    if not p.exists():
        print(f"  export not found: {p}")
        return 1
    msgs, _ = wa_study.parse(p)
    msgs = [m for m in msgs if m["text"].strip()][:a.cap]
    print(f"  {a.chat}: {len(msgs)} messages, chunks of {a.chunk}")

    chunks = [msgs[i:i + a.chunk] for i in range(0, len(msgs), a.chunk)]
    chunks = [c for c in chunks if len(c) >= 6]
    jobs = [(ci, k) for ci, c in enumerate(chunks) for k in range(len(c))]

    def one(job):
        ci, k = job
        c = chunks[ci]
        w = window([{"sender": m["who"], "text": m["text"]} for m in c], k)
        return ci, k, ask(key, a.model, w)

    labels, done = {}, 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for ci, k, v in ex.map(one, jobs):
            if v.startswith("ERR"):
                print(f"  API error {v} - stopping"); return 1
            labels[(ci, k)] = v
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)}")

    out, fires = [], Counter()
    for ci, c in enumerate(chunks):
        turns = []
        for k, m in enumerate(c):
            v = labels[(ci, k)]
            if v != "none":
                fires[v] += 1
            turns.append({"sender": m["who"], "text": m["text"],
                          "fire": [v] if v != "none" else []})
        out.append({"room": f"{a.chat}#{ci}", "source": a.chat,
                    "expects_fire": any(t["fire"] for t in turns), "turns": turns})
    (ROOT / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    n = len(jobs)
    print(f"\n  {n} messages, {len(out)} conversations")
    print(f"  teacher fires on {sum(fires.values())} = {sum(fires.values())/n*1000:.1f} per 1000")
    print(f"  {dict(fires)}")
    print("\n  what it fired on (these are the cases to check):\n")
    shown = 0
    for ci, c in enumerate(chunks):
        for k, m in enumerate(c):
            if labels[(ci, k)] != "none" and shown < 18:
                shown += 1
                prev = c[k - 1]["text"][:46] if k else ""
                print(f"    [{labels[(ci,k)]:5}] prev: {prev!r}")
                print(f"             >>  {m['text'][:62]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
