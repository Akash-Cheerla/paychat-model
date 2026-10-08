"""Ask DeepSeek for split-thread SCENARIOS. We compute the right answer ourselves.

Akash, 2026-10-07: "test this case as thoroughly as possible with multiple different
scenarios and stuff, if needed get some test cases from deepseek".

The division of labour matters. DeepSeek is good at inventing realistic group-chat
threads and bad at being an oracle, so it is asked ONLY for the thread plus the handful of
facts needed to score it (what the bill was, how many people are splitting, which message
is the commitment, how many shares that person is paying). The expected amount is then
computed here from FIRING_RULE.md 5a/5a1/5a3 - never taken from the model.

Everything sent is synthetic text written in this file. No chat log, no export and no
dogfood data goes to the API. Nothing that comes back is training data; these are test
cases only, consistent with the 2026-09-10 measurement-only approval for DeepSeek.

    python data_gen/gen_split_scenarios.py --n 120 --out data/eval/split_scenarios_llm.json
    python tests/test_split_matrix.py --url ... --cases data/eval/split_scenarios_llm.json
"""
import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEM = """You write realistic ENGLISH group-chat threads where one person paid a bill
and asks the others for their share. We are testing a payment-prompt system, so we need
awkward, messy, real threads - not clean textbook ones.

Return ONLY JSON: {"scenarios":[{...}]} where each scenario is:
{
  "name": "<6-10 word description of what makes this thread tricky>",
  "participants": <how many people are in the room, or null to say the room size is unknown>,
  "total_value": <the bill as a plain number, e.g. 208 or 95.5>,
  "total_text": "<the bill exactly as it is written in the message, e.g. \\"$208\\" or \\"Rs 500\\" or \\"90 rupees\\">",
  "currency": "USD" | "INR" | "EUR" | "bare",
  "headcount": <how many people the bill is ACTUALLY split between, or null if the thread
                never makes it knowable>,
  "payer_shares": <how many people's shares the committing person is paying - usually 1>,
  "messages": [{"sender": "<one of 10,20,30,40,50>", "text": "..."}]
}

Rules for the threads:
- The LAST message must be the one person committing to pay ("sending my share", "ok
  sending", "sending now"). Use those plain wordings - other phrasings are being measured
  separately and would confuse this test.
- The bill must be stated as a TOTAL for the group, not as a per-person figure, unless the
  scenario is specifically about a per-person figure (then set headcount to null and make
  "total_value" the per-person figure).
- "total_text" must appear verbatim inside one of the messages.
- If a later message changes the bill or the headcount, the LAST such statement is what
  "total_value" and "headcount" must reflect.
- headcount is null when the thread genuinely does not say: no room size given, or a vague
  count like "4 or 5 of us", or "some of us".
- Use different senders. The asker is whoever states the bill.
- No Hinglish, no transliteration, no emoji-only messages. English only.

Make them varied and difficult: clarifying questions, someone refusing, the total being
corrected, people dropping out, the headcount arriving several messages later, chit-chat in
between, a second unrelated bill, decimals, large bills, tiny bills, uneven divisions,
different currencies, vague counts.
"""

USER = """Write {k} scenarios. Make every one structurally different from the others.
At least a third should have the headcount or the total arrive AFTER the initial ask.
Include a few where headcount is null."""


def expected_share(value, n, cur):
    """FIRING_RULE 5a1. Mirrors tests/test_split_matrix.expected_share."""
    if not n or n < 2:
        return None
    share = value / n
    if share == int(share):
        return str(int(share))
    if cur == "INR":
        # Half rounds DOWN - see the comment in app.py's _per_person_share.
        r = math.ceil(share - 0.5)
        return str(r) if r else None
    txt = f"{share:.2f}"
    return txt if float(txt) else None


# How the share comes back for each way of writing the bill. A bare figure is marked "$".
def render(total_text, share_txt, cur):
    if share_txt is None:
        return None
    t = total_text.strip()
    num = re.search(r"\d[\d,]*(?:\.\d+)?", t)
    if not num:
        return None
    body = (t[:num.start()] + "\x00" + t[num.end():])
    out = body.replace("\x00", share_txt).strip()
    if cur == "INR":
        # "Rs"/"Rs."/"INR"/"rupees" - the slot extractor normalises the first three to the
        # symbol and leaves the trailing word alone.
        if re.match(r"^(rs\.?|inr)\b", t, re.IGNORECASE):
            return "₹" + share_txt
        return out
    if cur == "bare" or not re.search(r"[^\d\s.,]", t):
        return "$" + share_txt
    return out


def validate(sc):
    """Drop a scenario DeepSeek got structurally wrong, with a reason."""
    for k in ("name", "total_value", "total_text", "currency", "messages"):
        if k not in sc:
            return f"missing {k}"
    msgs = sc.get("messages") or []
    if len(msgs) < 2:
        return "needs at least 2 messages"
    if not all(isinstance(m, dict) and m.get("text") and m.get("sender") for m in msgs):
        return "malformed message"
    if sc["currency"] not in ("USD", "INR", "EUR", "bare"):
        return f"unknown currency {sc['currency']!r}"
    if not any(str(sc["total_text"]).strip() in (m["text"] or "") for m in msgs):
        return f"total_text {sc['total_text']!r} is not in any message"
    try:
        float(sc["total_value"])
    except (TypeError, ValueError):
        return "total_value is not a number"
    n = sc.get("headcount")
    if n is not None and (not isinstance(n, int) or n < 1 or n > 50):
        return f"implausible headcount {n!r}"
    p = sc.get("payer_shares", 1)
    if not isinstance(p, int) or p < 1 or p > 20:
        return f"implausible payer_shares {p!r}"
    if n and p > n:
        return f"payer_shares {p} exceeds headcount {n}"
    if re.search(r"[ऀ-ॿ]", json.dumps(sc, ensure_ascii=False)):
        return "non-English script"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out", default="data/eval/split_scenarios_llm.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEEPSEEK_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        print("no DEEPSEEK_API_KEY in the environment or .env")
        return 2

    from openai import OpenAI
    ds = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
    random.seed(a.seed)

    raw, dropped = [], []
    while len(raw) < a.n:
        k = min(a.batch, a.n - len(raw))
        try:
            r = ds.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER.format(k=k)}],
                temperature=1.0, max_tokens=8192,
                response_format={"type": "json_object"})
            got = json.loads(r.choices[0].message.content).get("scenarios", [])
        except Exception as e:                      # noqa: BLE001
            print(f"  batch failed ({type(e).__name__}: {e}), stopping early")
            break
        if not got:
            print("  empty batch, stopping early")
            break
        for sc in got:
            why = validate(sc)
            if why:
                dropped.append((sc.get("name", "?"), why))
            else:
                raw.append(sc)
        print(f"  {len(raw)}/{a.n} kept, {len(dropped)} dropped", flush=True)

    cases = []
    for sc in raw:
        share = expected_share(float(sc["total_value"]), sc.get("headcount"), sc["currency"])
        shares = sc.get("payer_shares", 1) or 1
        if share is not None and shares > 1:
            v = float(share) * shares
            share = str(int(v)) if v == int(v) else f"{v:.2f}"
        cases.append({
            "name": "llm: " + str(sc["name"])[:70],
            "participants": sc.get("participants"),
            "messages": [{"sender": str(m["sender"]), "text": m["text"]} for m in sc["messages"]],
            "expected": render(sc["total_text"], share, sc["currency"]),
            "_facts": {k: sc.get(k) for k in
                       ("total_value", "total_text", "currency", "headcount", "payer_shares")},
        })

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n  {len(cases)} scenarios -> {a.out}")
    print(f"  {sum(1 for c in cases if c['expected'] is None)} expect a BLANK amount")
    if dropped:
        print(f"\n  dropped {len(dropped)}:")
        for name, why in dropped[:15]:
            print(f"    {why}  <- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
