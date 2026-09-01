"""What does an LLM score on the same test set, with no training at all?

Six training rounds have failed to beat v5, and every explanation has been about data or
recipe. Nobody has checked the obvious alternative: hand the firing rules and the
conversation window to a general model and ask it.

Same window the classifier sees - last N messages, speakers normalised - so the
comparison is fair. Same scoring: fires when it should, stays quiet when it should.

    python llm_baseline.py --n 60 --out data/eval/LLM_baseline.json
"""
import argparse, json, os, re, sys, io, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
API = "https://api.deepseek.com/chat/completions"

RULES = """You decide whether a chat app should show a payment or ride-booking prompt
on the LAST message of a conversation.

FIRES:
- Someone asks for money or a cab, and another person AGREES ("ok", "sure", "yeah",
  "k", "of course", "will do", "why not", a thumbs up).
- Someone states an action already in progress about themselves: "im booking an uber",
  "sending you 20 now", "on it".
- An offer that ANSWERS a request: "can you book me a cab" / "let me book a cab".
- An unprompted offer AFTER the other person confirms it: "let me send you 500" / "ok".

DOES NOT FIRE:
- A request on its own, with nobody agreeing yet.
- A need, not an action: "i'll need to book a cab", "i may book a cab later".
- An unprompted offer nobody has answered yet: a bare "let me book a cab".
- A future promise: "i'll pay you back tomorrow".
- Something already done: "just sent it", "cab booked", "already paid".
- Preparation: "one sec opening the app", "lemme grab my phone".
- A greeting, even right after a request: "hi", "hey", "yo".
- A rejection: "no", "cant", "not now".
- People DISCUSSING the app, quoting example conversations, or debugging it.
- Ride means a hired cab/uber/ola only, never a friend's own car or a bus.
- Money means one person paying another person now, not bills or reminders.

Answer with exactly one word: money, ride, or none."""


def window(turns, k, size=10):
    lo = max(0, k - size + 1)
    sub = turns[lo:k + 1]
    last = str(sub[-1]["sender"])
    names, out = {last: "A"}, []
    nxt = iter("BCDEFGH")
    for t in sub:
        s = str(t["sender"])
        if s not in names:
            names[s] = next(nxt)
        out.append(f"{names[s]}: {t['text']}")
    return "\n".join(out)


def ask(key, text, retries=4):
    for i in range(retries):
        try:
            r = requests.post(API, timeout=90, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "deepseek-chat", "temperature": 0, "max_tokens": 4,
                      "messages": [{"role": "system", "content": RULES},
                                   {"role": "user", "content": text}]})
            if r.status_code == 200:
                w = r.json()["choices"][0]["message"]["content"].strip().lower()
                return "money" if "money" in w else ("ride" if "ride" in w else "none")
            if r.status_code in (402, 401):
                return f"ERR{r.status_code}"
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return "ERR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="data/eval/LLM_baseline.json")
    ap.add_argument("--ids", default="")
    a = ap.parse_args()

    key = ""
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK_API_KEY"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
    assert key, "no DEEPSEEK_API_KEY in .env"

    convs = json.loads((ROOT / "data/eval/human_eval.json").read_text(encoding="utf-8"))
    if a.ids:
        idx = json.loads((ROOT / a.ids).read_text(encoding="utf-8"))[:a.n]
    else:
        fire = [i for i, c in enumerate(convs) if c["expects_fire"]][:a.n // 2]
        quiet = [i for i, c in enumerate(convs) if not c["expects_fire"]][:a.n // 2]
        idx = fire + quiet

    def one(ci):
        c = convs[ci]
        want = next((t["fire"][0] for t in c["turns"] if t.get("fire")), None)
        got_any, got_right = False, False
        # Ask on every turn, exactly like the classifier is asked on every message.
        for k in range(len(c["turns"])):
            v = ask(key, window(c["turns"], k))
            if v.startswith("ERR"):
                return ci, None, v
            if v != "none":
                got_any = True
                if v == want:
                    got_right = True
        return ci, (got_any, got_right), None

    res = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for ci, out, err in ex.map(one, idx):
            if err:
                print(f"  API error: {err}"); return 1
            res[ci] = out

    fo = fn = qo = qn = 0
    for ci in idx:
        c = convs[ci]
        got_any, got_right = res[ci]
        if c["expects_fire"]:
            fn += 1; fo += got_right
        else:
            qn += 1; qo += (not got_any)
    print(f"\n  LLM with the rules in the prompt, no training, {len(idx)} conversations")
    print(f"    FIRES WHEN IT SHOULD   {fo}/{fn} = {fo/max(fn,1)*100:5.1f}%")
    print(f"    STAYS QUIET            {qo}/{qn} = {qo/max(qn,1)*100:5.1f}%")
    print(f"    overall                {(fo+qo)}/{fn+qn} = {(fo+qo)/(fn+qn)*100:5.1f}%")
    (ROOT / a.out).write_text(json.dumps(
        {"ids": idx, "fire": [fo, fn], "quiet": [qo, qn]}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
