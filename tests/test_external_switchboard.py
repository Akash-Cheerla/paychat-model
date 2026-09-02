"""Run the model over real human conversation nobody here wrote.

Every number this repo has produced came from data we generated ourselves. That is the
weakness that made six retrains unmeasurable: a model tuned on our own conversations,
scored on our own conversations, tells you nothing about conversations.

The Switchboard Dialog Act Corpus is 1,155 telephone conversations between American
strangers, recorded 1990-91 for linguistics research. It has nothing to do with this
product, was annotated by people who have never heard of it, and predates the phrases
our training data is built from by thirty years.

It cannot test whether we fire CORRECTLY - nobody in it is sending money over Venmo or
booking an Ola. What it tests is the thing that actually costs us:

    does the model stay silent on ordinary human conversation?

A prompt here is a false fire on real speech. The only defensible result is close to
zero. Anything else is the model reacting to conversational surface - "sure", "okay",
"I'll do that" - rather than to money and rides, and no internal battery can show that
because we wrote the internal batteries.

Switchboard markup is stripped before scoring: {F uh}, {D well}, [ x + y ] repairs,
<Noise>, and the trailing / that marks a slash-unit. What is left is what a person said.

    python tests/test_external_switchboard.py --swda path/to/swda.zip \
                                              --url http://127.0.0.1:8900/detect
"""
import argparse, csv, io, json, re, sys, time, zipfile
from collections import Counter
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
MR = ("money", "ride")

# Switchboard transcription conventions, in the order they have to come off
_CURLY = re.compile(r"\{[A-Z]\s*")          # {F uh,  {D well,  {C and
_REPAIR = re.compile(r"[\[\]\+]")            # [ it, + it ]
_ANGLE = re.compile(r"<+[^>]*>+")            # <Noise>, <<very faint>>
_ASIDE = re.compile(r"[#()]")
_SLASH = re.compile(r"\s*/\s*$")
_SPACE = re.compile(r"\s+")


def clean(t: str) -> str:
    t = _ANGLE.sub(" ", t or "")
    t = _CURLY.sub(" ", t)
    t = _REPAIR.sub(" ", t)
    t = _ASIDE.sub(" ", t)
    t = _SLASH.sub("", t)
    t = t.replace("}", " ")
    t = _SPACE.sub(" ", t).strip()
    # a turn that was only disfluency markup leaves nothing worth sending
    return t if len(t) > 1 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swda", required=True, help="swda.zip from github.com/cgpotts/swda")
    ap.add_argument("--url", required=True)
    ap.add_argument("--conversations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()

    z = zipfile.ZipFile(a.swda)
    names = sorted(n for n in z.namelist()
                   if n.endswith(".csv") and not n.startswith("__MACOSX"))
    import random
    random.Random(a.seed).shuffle(names)
    names = names[:a.conversations]

    tag = int(time.time())
    total_msgs = fires = 0
    hits, by_conv = [], Counter()

    for ci, n in enumerate(names):
        rows = list(csv.DictReader(io.StringIO(
            z.read(n).decode("utf-8", errors="replace"))))
        room = f"swda_{tag}_{ci}"
        turns = []
        for r in rows:
            t = clean(r.get("text", ""))
            if t:
                turns.append((r.get("caller", "A"), t))
        for ti, (spk, txt) in enumerate(turns):
            try:
                d = requests.post(a.url, timeout=60, json={
                    "text": txt, "room_id": room, "sender": spk,
                    "message_id": f"{tag}_{ci}_{ti}"}).json()
            except Exception:
                continue
            total_msgs += 1
            got = [x for x in (d.get("intents") or []) if x in MR]
            if got:
                fires += 1
                by_conv[n] += 1
                sc = (d.get("conversation_state") or {}).get("scores") or {}
                prev = turns[ti - 1][1] if ti else ""
                hits.append({"conv": n.split("/")[-1], "i": ti, "intents": got,
                             "text": txt, "prev": prev,
                             "score": round(max(sc.get("money", 0), sc.get("ride", 0)), 3)})
        if (ci + 1) % 10 == 0:
            print(f"    {ci+1}/{len(names)} conversations, {total_msgs:,} messages, {fires} fires")

    print(f"\n  === Switchboard, {len(names)} conversations ===")
    print(f"  messages scored : {total_msgs:,}")
    print(f"  prompts fired   : {fires}")
    print(f"  rate            : {fires/max(total_msgs,1)*1000:.2f} per 1,000 messages")
    print(f"  conversations with at least one prompt: {len(by_conv)} of {len(names)}")

    if hits:
        print(f"\n  every prompt, with the line before it:")
        for h in hits[:30]:
            print(f"    [{h['conv']}#{h['i']}] {h['intents']} {h['score']}")
            if h["prev"]:
                print(f"        prev: {h['prev'][:72]!r}")
            print(f"        ---> {h['text'][:72]!r}")
    if a.save:
        Path(a.save).write_text(json.dumps(hits, indent=1, ensure_ascii=False),
                                encoding="utf-8")
        print(f"\n  wrote {a.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
