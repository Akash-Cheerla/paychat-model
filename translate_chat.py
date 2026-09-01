"""Translate a real chat into English, keeping it a chat.

A quarter of the Anupam export is Telugu written in English letters, and the money and
ride moments sit disproportionately in that quarter. So we cannot tell whether the model
is failing on those or simply cannot read them. Translating first separates the two
questions: run the English version and whatever it still misses is the model's fault.

Translated in order and in batches, because the classifier reads a ten-message window -
a chat half translated and half not is a window that never existed.

The instruction is to keep it sounding like texting. "Paisal phone pe chesta amount
chepu" must come out as "i'll send the money on phonepe, tell me the amount", not as
"Kindly inform me of the amount and I shall transfer it via PhonePe." A polished
translation would test a register none of our users write in.

Output is gitignored: it is derived from private messages and carries their restrictions.
"""
import argparse, json, re, sys, io, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data_gen"))
import wa_study                                    # noqa: E402

sys.stdout = io.TextIOWrapper(open(sys.__stdout__.fileno(), "wb", closefd=False),
                              encoding="utf-8", errors="replace", line_buffering=True)
API = "https://api.deepseek.com/chat/completions"

SYSTEM = """You translate group-chat messages into casual English.

The messages are a mix of English and Telugu or Hindi written in English letters. Some
are already English - return those unchanged.

Rules:
- Keep it sounding like texting. Lowercase, short, contractions, slang. Never formal.
- "Paisal phone pe chesta amount chepu" -> "i'll send the money on phonepe, tell me the amount"
- Keep names, numbers, amounts, places and emoji exactly as they are.
- Keep attachment markers like <attached: ...> and <link> exactly as they are.
- One line in, one line out. Never merge, split, explain or add anything.

You get numbered lines. Return the same numbers, same count, nothing else."""


def translate(key, model, batch, retries=4):
    body = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
    for attempt in range(retries):
        try:
            r = requests.post(API, timeout=180, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "temperature": 0,
                      "messages": [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": body}]})
            if r.status_code in (401, 402):
                return None
            if r.status_code == 200:
                out = r.json()["choices"][0]["message"]["content"]
                got = {}
                for line in out.splitlines():
                    m = re.match(r"\s*(\d+)\s*[.)]\s?(.*)$", line)
                    if m:
                        got[int(m.group(1))] = m.group(2).strip()
                # only accept a batch that came back whole; a short one would
                # silently shift every later message onto the wrong turn
                if len(got) == len(batch):
                    return [got[i + 1] for i in range(len(batch))]
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True, help="folder under data/private_wa")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--batch", type=int, default=20)
    a = ap.parse_args()

    key = ""
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DEEPSEEK_API_KEY"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
    assert key, "no DEEPSEEK_API_KEY"

    msgs, _ = wa_study.parse(ROOT / f"data/private_wa/{a.chat}/_chat.txt")
    msgs = [m for m in msgs if m["text"].strip()]
    print(f"  {a.chat}: {len(msgs)} messages, batches of {a.batch}")

    idx = list(range(0, len(msgs), a.batch))
    batches = [[m["text"] for m in msgs[i:i + a.batch]] for i in idx]

    def one(k):
        return k, translate(key, a.model, batches[k])

    out = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, res in ex.map(one, range(len(batches))):
            if res is None:
                print(f"  batch {k} failed - keeping the original text for it")
                res = batches[k]
            out[k] = res

    rows = []
    changed = 0
    for k, start in enumerate(idx):
        for j, t in enumerate(out[k]):
            m = msgs[start + j]
            if t.strip().lower() != m["text"].strip().lower():
                changed += 1
            rows.append({"who": m["who"], "text": t, "original": m["text"]})
    (ROOT / a.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print(f"  wrote {a.out}: {len(rows)} messages, {changed} changed "
          f"({changed/len(rows)*100:.0f}%)")
    print("\n  sample of what changed:\n")
    shown = 0
    for r in rows:
        if r["text"].strip().lower() != r["original"].strip().lower() and shown < 10:
            shown += 1
            print(f"    {r['original'][:56]}")
            print(f"      -> {r['text'][:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
