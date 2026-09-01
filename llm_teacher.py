"""Is an LLM accurate enough to LABEL data for us?

Not a runtime component - it never goes near the app. The question is whether it can
label real conversations well enough to build a test set and a training set from, because
that is the thing actually blocking us: every label our models have ever seen came from
4-to-10-turn synthetic conversations, and real chats run to 15-40 turns.

A first pass with the rules alone and no examples scored 73.3 / 90.0 on 60 conversations,
against v5's 80.0 / 90.0 on the same ones. This adds worked examples for the cases we
have actually confirmed as failures - greetings after a request, quoted conversations,
needs vs offers vs actions in progress, reluctant agreement - and reports where it
disagrees with our labels, so the disagreements can be read rather than trusted.

    python llm_teacher.py --n 120 --out data/eval/LLM_teacher.json
"""
import argparse, json, sys, io, time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
API = "https://api.deepseek.com/chat/completions"

SYSTEM = """You decide whether a chat app should show a payment or ride-booking prompt on
the LAST message shown. Answer with exactly one word: money, ride, or none.

The prompt exists so someone can complete an action. It fires the moment a person has
committed to sending money or booking a cab - not before, not again afterwards.

FIRES
- A request, then someone else AGREES: "ok" / "sure" / "yeah" / "k" / "of course" /
  "will do" / "why not" / a thumbs up. The agreement fires it, however short.
  THE ANSWER IS WHATEVER WAS ASKED FOR. If a cab was asked for, the agreement is a
  ride. If money was asked for, it is money. Never guess money by default - look back
  and see what the request actually was. If both were asked for, answer the one asked
  most recently.
- Agreement that arrives late, after the conversation drifted to other topics.
- Reluctant agreement: "ugh fine", "whatever i'll do it", "fine i'll book it".
- An action stated as in progress about oneself: "im booking an uber", "sending you 20
  now", "on it", "booking now".
- An offer that ANSWERS a request: "can you book me a cab" -> "let me book a cab".
- An unprompted offer once the other person confirms: "let me send you 500" -> "ok".

DOES NOT FIRE
- A request by itself, with nobody agreeing yet.
- A need rather than an action: "i'll need a cab later", "i may book a cab".
- An unprompted offer nobody has answered: a bare "let me book a cab".
- A future promise: "i'll pay you back tomorrow", "will send next week".
- Something already finished: "just sent it", "cab booked", "already paid".
- Preparation: "one sec opening the app", "lemme grab my phone".
- A greeting, even directly after a request: "hi", "hey", "yo", "good morning".
- A rejection: "no", "can't", "not now", "maybe later".
- People DISCUSSING this app, quoting example conversations, or debugging it. Text that
  quotes a conversation is not a conversation.
- Ride means a hired cab / uber / ola only. Never a friend's own car, a bus, a train,
  or a flight.
- Money means one person paying another person now. Not bills, not reminders, not
  splitting a plan nobody has agreed to.
- FOOD IS NOT A RIDE AND NOT A PAYMENT. Ordering food, placing a delivery order, or
  agreeing to order food fires NOTHING. "ordering now", "ok ordering", "placing the
  order" are about food unless the conversation is already about a cab. Judge by what
  the conversation is about, not by whether the agreeing message repeats the word: if
  someone asked for a cab and a person agrees, that is a ride even if they only say
  "fine i'll do it".
- Deciding who will pay later, or that someone will be paid back, is not a payment now."""

SHOTS = [
    ("B: can you send me 10$\nA: Hi", "none"),
    ("B: can you send me 10$\nA: ok", "money"),
    ("B: can you book cab for me\nA: Sure", "ride"),
    ("B: Book a cab from my location to marathalli\nA: Sure", "ride"),
    ("B: Set a reminder tomorrow 10am\nB: Book a cab from my location\nA: Sure", "ride"),
    ("B: Can you send me 10 rupees\nB: Can you book cab for me\nA: Sure", "ride"),
    ("B: hey\nA: let me book a cab", "none"),
    ("B: can you book me a cab to the airport\nA: let me book a cab", "ride"),
    ("B: hey\nB: yo\nA: im booking an uber", "ride"),
    ("A: i'll need to book a cab ride back home", "none"),
    ("B: How long shall we keep the request open?\nB: Let's say\nB: can you book me a "
     "cab to airport\nB: A: okay\nA: Yes", "none"),
    ("B: can you spot me 500\nA: just sent it", "none"),
    ("B: can you lend me 2000\nA: for what\nB: car broke down\nA: lol true\nB: so yeah\n"
     "A: fine i'll do it", "money"),
    ("B: can you book me a cab\nA: one sec opening the app", "none"),
    ("B: shall we get pizza\nA: ok ordering now", "none"),
    ("B: im hungry\nB: order something\nA: ordering now", "none"),
    ("B: whos paying for the food\nA: ill get it, pay me back whenever", "none"),
    ("C: cab to indiranagar anyone\nB: cant, stuck in meeting\nD: same lol\n"
     "A: fine i'll do it", "ride"),
    ("B: anyone can book cab to airport saturday?\nC: i hate airports\nB: 9am flight\n"
     "C: ok whos bringing snacks\nA: yeah fine i can do it", "ride"),
]


def window(turns, k, size=10):
    sub = turns[max(0, k - size + 1):k + 1]
    last = str(sub[-1]["sender"])
    names, out, nxt = {last: "A"}, [], iter("BCDEFGH")
    for t in sub:
        s = str(t["sender"])
        if s not in names:
            names[s] = next(nxt)
        out.append(f"{names[s]}: {t['text']}")
    return "\n".join(out)


def ask(key, model, text, retries=4):
    msgs = [{"role": "system", "content": SYSTEM}]
    for w, a in SHOTS:
        msgs.append({"role": "user", "content": w})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": text})
    for i in range(retries):
        try:
            r = requests.post(API, timeout=90, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "temperature": 0, "max_tokens": 4,
                      "messages": msgs})
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
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many of each class first; use a non-zero value to "
                         "score on conversations the prompt was not tuned against")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--out", default="data/eval/LLM_teacher.json")
    a = ap.parse_args()

    key = ""
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK_API_KEY"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
    assert key, "no DEEPSEEK_API_KEY"

    convs = json.loads((ROOT / "data/eval/human_eval.json").read_text(encoding="utf-8"))
    o = a.offset
    fire = [i for i, c in enumerate(convs) if c["expects_fire"]][o:o + a.n // 2]
    quiet = [i for i, c in enumerate(convs) if not c["expects_fire"]][o:o + a.n // 2]
    idx = fire + quiet

    def one(ci):
        c = convs[ci]
        want = next((t["fire"][0] for t in c["turns"] if t.get("fire")), None)
        got_any = got_right = False
        first = None
        for k in range(len(c["turns"])):
            v = ask(key, a.model, window(c["turns"], k))
            if v.startswith("ERR"):
                return ci, None
            if v != "none":
                got_any = True
                if first is None:
                    first = (k, v, c["turns"][k]["text"][:48])
                if v == want:
                    got_right = True
        return ci, {"scenario": c["scenario"], "expects_fire": bool(c["expects_fire"]),
                    "want": want, "fired": got_any, "right": got_right, "first": first}

    res = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for ci, out in ex.map(one, idx):
            if out is None:
                print("  API error - stopping"); return 1
            res[ci] = out

    fo = fn = qo = qn = 0
    by = Counter()
    for ci, r in res.items():
        if r["expects_fire"]:
            fn += 1; fo += r["right"]
            if not r["right"]: by[("MISS", r["scenario"])] += 1
        else:
            qn += 1; qo += (not r["fired"])
            if r["fired"]: by[("FALSE", r["scenario"])] += 1
    print(f"\n  LLM as teacher ({a.model}, {len(SHOTS)} worked examples), {len(idx)} conversations")
    print(f"    FIRES WHEN IT SHOULD   {fo}/{fn} = {fo/max(fn,1)*100:5.1f}%")
    print(f"    STAYS QUIET            {qo}/{qn} = {qo/max(qn,1)*100:5.1f}%")
    print(f"    overall                {fo+qo}/{fn+qn} = {(fo+qo)/(fn+qn)*100:5.1f}%")
    print("\n  where it disagrees with our labels:")
    for (kind, scen), n in by.most_common(12):
        print(f"    {kind:6} {scen:26} {n}")
    (ROOT / a.out).write_text(json.dumps(
        {"model": a.model, "ids": idx, "fire": [fo, fn], "quiet": [qo, qn],
         "detail": {str(k): v for k, v in res.items()}}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
