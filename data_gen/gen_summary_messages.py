"""Ask DeepSeek for long team-chat messages, with what MUST survive the summary marked.

Akash, 2026-10-08: "summary is to summarize important parts of messages not everything".
That is a recall-and-precision question and fourteen hand-written cases cannot answer it,
so the messages are generated and each one carries two lists:

    must_keep   the things a reader would be annoyed to lose - a deadline, an owner, a
                number, a decision, something broken
    must_drop   the chatter around them - greetings, opinions, hedging, sign-offs

Those two lists are what the test scores against, and unlike the split work they are NOT
arithmetic, so they are a judgement the model is making. They are checked two ways before
they are trusted: every string must appear verbatim in the message, and a must_keep and a
must_drop may not overlap. Anything else is spot-checked by eye - see the note in
tests/test_summary_messages.py about what the generated labels are and are not good for.

Everything sent is synthetic text written in this file. No chat log, no export and no
dogfood data goes to the API, and nothing that comes back is training data - these are
test cases only, consistent with the 2026-09-10 measurement-only approval for DeepSeek.

    python data_gen/gen_summary_messages.py --n 90 --out data/eval/summary_messages.json
"""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEM = """You write realistic ENGLISH messages from a software team's group chat - the
long ones, the kind someone types after a day of testing or before a release. They are
being used to test a feature that turns a long message into a few bullet points, so they
must be messy and human: greetings, tangents, opinions, half-finished thoughts mixed in
with the things that actually matter.

Return ONLY JSON: {"messages":[{...}]} where each is:
{
  "name": "<6-10 words saying what makes this message tricky>",
  "text": "<the message, 5 to 15 sentences, one paragraph or several>",
  "must_keep": ["<verbatim substring>", ...],
  "must_drop": ["<verbatim substring>", ...]
}

must_keep: the parts a teammate would be annoyed to lose when they scroll back a day
later. Deadlines and dates, who owns what, amounts and counts, decisions that were taken,
anything broken or blocking, explicit instructions. Quote them EXACTLY as they appear in
the text, a few words each, enough to identify the sentence.

must_drop: the parts that are only noise in a summary. Greetings and sign-offs, "hope
you're all well", opinions ("I feel like", "honestly"), hedging ("probably", "I think"),
filler ("anyway", "nothing major"), thanks. Quote these EXACTLY as they appear too.

Rules:
- Every must_keep and must_drop string must appear in "text" character for character.
- A must_keep and a must_drop must never overlap in the text.
- At least 2 must_keep and at least 2 must_drop per message.
- Some messages should be ALL chatter with nothing worth keeping: then must_keep is []
  and the whole thing is must_drop.
- English only. No Hinglish, no transliteration.
- Vary it: bug reports, release notes, QA findings, design review, a deadline chase, a
  post-mortem, someone rambling, a message that is mostly a list, a message with links.
"""

USER = """Write {k} messages. Make each one structurally different. Include at least two
that are entirely chatter with nothing worth summarizing."""


def validate(m):
    for k in ("name", "text", "must_keep", "must_drop"):
        if k not in m:
            return f"missing {k}"
    text = m["text"]
    if not isinstance(text, str) or len(text) < 140:
        return "text too short to be summarised at all"
    for key in ("must_keep", "must_drop"):
        if not isinstance(m[key], list):
            return f"{key} is not a list"
        for s in m[key]:
            if not isinstance(s, str) or not s.strip():
                return f"{key} has an empty entry"
            if s not in text:
                return f"{key} entry is not verbatim in the text: {s[:40]!r}"
    if len(m["must_drop"]) < 2:
        return "needs at least 2 must_drop"
    # An overlapping pair makes the message unscorable: the same words cannot be both
    # required and forbidden.
    for a in m["must_keep"]:
        for b in m["must_drop"]:
            if a in b or b in a:
                return f"must_keep and must_drop overlap: {a[:30]!r} / {b[:30]!r}"
    if re.search(r"[ऀ-ॿ]", text):
        return "non-English script"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--out", default="data/eval/summary_messages.json")
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

    kept, dropped = [], []
    while len(kept) < a.n:
        k = min(a.batch, a.n - len(kept))
        try:
            r = ds.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER.format(k=k)}],
                temperature=1.0, max_tokens=8192,
                response_format={"type": "json_object"})
            got = json.loads(r.choices[0].message.content).get("messages", [])
        except Exception as e:                      # noqa: BLE001
            print(f"  batch failed ({type(e).__name__}: {e}), stopping early")
            break
        if not got:
            print("  empty batch, stopping early")
            break
        for m in got:
            why = validate(m)
            (dropped if why else kept).append((m.get("name", "?"), why) if why else m)
        print(f"  {len(kept)}/{a.n} kept, {len(dropped)} dropped", flush=True)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n  {len(kept)} messages -> {a.out}")
    print(f"  {sum(1 for m in kept if not m['must_keep'])} are pure chatter (expect NO card)")
    print(f"  {sum(len(m['must_keep']) for m in kept)} must_keep, "
          f"{sum(len(m['must_drop']) for m in kept)} must_drop in all")
    if dropped:
        print(f"\n  dropped {len(dropped)}:")
        for name, why in dropped[:12]:
            print(f"    {why}  <- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
