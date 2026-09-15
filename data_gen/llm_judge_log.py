"""Judge every message of a real dogfood log with an LLM, on the full rules.

The classifier has plateaued: v6-v16 each fixed about three test cases and broke about
three. The one thing that has measurably beaten it is an LLM holding the full FIRING_RULE
- 79/82 on the context/group cases against v14's 75, and on real traffic 98/125 disputed
messages (09-10) and 44/58 on a blind holdout (08-11) against the live model's 82 and 33.

Every non-test message is judged in context - up to 14 messages before it in the same
room - with the rules as a constant system message, so hosts that cache prefixes charge
almost nothing for them. Test rooms are filtered by the same TESTP list replay_ab uses, so
the judged set is exactly the replayed set. compare_judge.py / score_judges.py line the
result up against the replayed models and the adjudicated answers.

--provider / --model run any OpenAI-compatible host, so open-weight models we could host
ourselves are measured on exactly the same prompt, messages and answers as DeepSeek:
    deepseek     the benchmark (DEEPSEEK_API_KEY)
    openrouter   hosted open models - gpt-oss, Qwen3, Gemma 3, Llama (OPENROUTER_API_KEY)
    hf           the HuggingFace router (HF_TOKEN)
    ollama       a model running on THIS machine - no key, nothing leaves the laptop

--prompt v2 adds three product rulings. It was tried and rejected: better on nothing, and
on the blind holdout it missed 26 of 32 prompts where v1 missed 11 - its "talk about the
app is not a commitment" ruling silences real commitments made in test rooms.
--stateful was tried and rejected: marking where sheets opened made the judge call the
FIRST acceptance of a re-sent request "already answered" - 46 misses on 09-10. Repeats
are the server's job; compare_judge's pending gate simulates it.

Real people's messages go to the provider for this - DeepSeek approved by Akash on
2026-09-10, for measurement. Any other host needs its own OK. Nothing here is ever
training data.

    python data_gen/llm_judge_log.py --log data/eval/dogfood_2026-09-10.jsonl
    python data_gen/llm_judge_log.py --log ... --provider openrouter --model openai/gpt-oss-20b
    python data_gen/llm_judge_log.py --log ... --provider ollama --model qwen3:8b
"""
import argparse, io, json, os, re, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
# replay_ab re-wraps stdout as UTF-8 when imported. Wrapping it here as well would leave
# the first wrapper unreferenced, and collecting it closes the stream under the second.
from replay_ab import load                                        # noqa: E402

MR = ("money", "ride")
CTX = 14

# Any OpenAI-compatible host. configure() switches the module, so callers that import
# ask() (eval_judge_82.py) follow without changes.
PROVIDERS = {
    "deepseek":   ("https://api.deepseek.com/chat/completions", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", None, "OPENROUTER_API_KEY"),
    "hf":         ("https://router.huggingface.co/v1/chat/completions", None, "HF_TOKEN"),
    "ollama":     ("http://localhost:11434/v1/chat/completions", None, None),
}
CFG = {"provider": "deepseek", "url": PROVIDERS["deepseek"][0], "model": "deepseek-chat",
       "key": "DEEPSEEK_API_KEY"}


def configure(provider="deepseek", model=None):
    url, default, keyname = PROVIDERS[provider]
    if not (model or default):
        raise SystemExit(f"  --model is required for {provider}")
    CFG.update(provider=provider, url=url, model=model or default, key=keyname)


def tag_for(provider, model):
    """File tag: '' only for the benchmark itself - deepseek-chat on DeepSeek - so its
    existing files keep their names. Every other model gets its own file, including
    DeepSeek's reasoner, which would otherwise have been appended to the benchmark's."""
    if provider == "deepseek" and model == "deepseek-chat":
        return ""
    return "_" + re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")


ADDENDUM = """
## R. A REQUEST ON ITS OWN NEVER FIRES  (product owner, 2026-08-14)

Not written above, but it governs all of it. A request - "book me a cab", "can you send
me 500" - opens a pending and shows NOTHING. The prompt appears when the responsible
party RESPONDS with a commitment. The only messages that fire with no preceding request
are self-initiated actions already in progress (SS3): "im sending you 20 now".
"""

SYSTEM = """You decide whether a chat app should open a payment sheet or a cab-booking
sheet at this moment. You get the product's complete firing rules, then the recent
conversation from a real chat room. Judge ONLY the final message, in the context of
everything before it.

Check these before answering - they are the mistakes judges make most on this task:

  FIRES  SS2   a terse acceptance of a REAL request: "sure", "ok", "done deal", "im on it",
               "count me in", a thumbs up
         SS3c  accepting an unprompted OFFER: "if you dont mind", "that would help", "please do"
         SS3   a self-initiated action IN PROGRESS: "sending now", "booking it now"
         SS6d  agreeing to send NOW with a condition about LATER: "ill send it now but pay
               me back friday"
         SS6e  the other person agreeing to a counter-offer: "i can only do 100" / "yeah that works"
  QUIET  SS3a  already done: "just sent it", "cab's booked"
         SS2   a future promise: "ill pay you friday", "i'll do mine in a bit"
         SS4   their own car, not ride-hailing: "ill drop u"
         SS1   a question: "should i send mine too?"
         -     a request on its own, or nothing real to accept

Reply with JSON and nothing else:
  {"fire": "money" | "ride" | "none", "rule": "<section>", "why": "<=12 words>"}"""

RULINGS = """

Three product rulings the rules below do not spell out:

  WHO   Only someone OTHER than the requester can accept a request. Find the most recent
        open request and who sent it. If the final line's sender made that request, their
        "sure" or "ok" is not an acceptance. People in one room often ask each other in turn.
  NOW   "i will send", "ok i will transfer", "i'll book it" with no later time is a
        commitment now. Only an explicit later time - "friday", "tonight", "in a bit",
        "after this meeting" - makes it a future promise.
  TALK  Talk ABOUT the app, its prompts or a test ("did you get the intent?", "testing
        this", a quoted example conversation) and jokes are not commitments."""

STATE_NOTE = """

Lines in [square brackets] are not chat. They are the app itself, recording that it
already opened a sheet at that point. A request answered by an opened sheet is finished:
a repeated "sure", "ok" or "sending now" about that SAME request does not open another
one. A NEW request after it starts over."""

USAGE = {"hit": 0, "miss": 0, "out": 0, "calls": 0, "cost": 0.0}
_ULOCK = threading.Lock()


def key():
    name = CFG["key"]
    if not name:
        return ""                                      # a local runner needs no key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    if os.environ.get(name):
        return os.environ[name]
    raise SystemExit(f"  no {name} in .env or the environment")


def system_prompt(stateful=False, v2=False):
    t = (ROOT / "FIRING_RULE.md").read_text(encoding="utf-8")
    i = t.find("## The one principle")
    return (f"{SYSTEM}{RULINGS if v2 else ''}{STATE_NOTE if stateful else ''}\n\n---\n\n"
            f"THE RULES:\n\n{t[i:] if i > 0 else t}\n\n{ADDENDUM}")


def parse(txt):
    """The LAST JSON object carrying "fire". Reasoning models (gpt-oss, Qwen3) may think
    out loud first, and that text can contain braces of its own."""
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    for c in reversed(re.findall(r"\{[^{}]*\}", txt, re.S)):
        try:
            d = json.loads(c)
            if "fire" in d:
                return d
        except Exception:
            continue
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0) if m else txt)


def ask(k, sysp, room, window, retries=4):
    kind = "a GROUP chat" if room.startswith("group") else "a one-to-one DM"
    convo = "\n".join(m["text"] if m.get("marker") else f"{m.get('sender')}: {m.get('text')}"
                      for m in window)
    last = window[-1]
    user = (f"Room type: {kind}.\n\nTHE CONVERSATION:\n\n{convo}\n\n"
            f"Judge ONLY the final line, sent by {last.get('sender')}: {last.get('text')!r}")
    body = {"model": CFG["model"], "temperature": 0,
            "messages": [{"role": "system", "content": sysp}, {"role": "user", "content": user}]}
    if CFG["provider"] != "deepseek":
        body["max_tokens"] = 1500                      # reasoning models, bounded
    if CFG["provider"] == "openrouter":
        body["usage"] = {"include": True}              # have it report the real cost
    headers = {"Content-Type": "application/json"}
    if k:
        headers["Authorization"] = f"Bearer {k}"
    if CFG["provider"] == "openrouter":
        headers["X-Title"] = "PayChat intent eval"
    for a in range(retries):
        try:
            r = requests.post(CFG["url"], timeout=600 if CFG["provider"] == "ollama" else 180,
                              headers=headers, json=body)
            if r.status_code in (401, 402, 403):
                return "AUTH", "", ""
            if r.status_code != 200:
                time.sleep(min(2 ** a, 30)); continue
            j = r.json()
            u = j.get("usage") or {}
            with _ULOCK:
                if "prompt_cache_hit_tokens" in u:     # DeepSeek's own accounting
                    USAGE["hit"] += int(u.get("prompt_cache_hit_tokens") or 0)
                    USAGE["miss"] += int(u.get("prompt_cache_miss_tokens") or 0)
                else:                                  # the OpenAI-standard fields
                    cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
                    USAGE["hit"] += cached
                    USAGE["miss"] += int(u.get("prompt_tokens") or 0) - cached
                USAGE["out"] += int(u.get("completion_tokens") or 0)
                USAGE["cost"] += float(u.get("cost") or 0)
                USAGE["calls"] += 1
            d = parse(j["choices"][0]["message"].get("content"))
            f = str(d.get("fire", "none")).lower()
            return (f if f in MR else None), str(d.get("rule", ""))[:10], str(d.get("why", ""))[:70]
        except Exception:
            time.sleep(min(2 ** a, 30))
    return "ERROR", "", ""


def usage_line():
    c = USAGE
    if not c["calls"]:
        return ""
    hitp = c["hit"] / max(c["hit"] + c["miss"], 1) * 100
    if c["cost"]:
        return (f"  API usage: {c['calls']:,} calls   cache hit {hitp:.1f}%   "
                f"reported cost ${c['cost']:.4f}")
    if CFG["provider"] == "deepseek":
        cost = c["hit"] / 1e6 * 0.003 + c["miss"] / 1e6 * 0.15 + c["out"] / 1e6 * 0.6
        return (f"  API usage: {c['calls']:,} calls   cache hit {hitp:.1f}%   "
                f"est. ${cost:.4f} off-peak (x2 at peak)")
    return f"  usage: {c['calls']:,} calls   {c['miss'] + c['hit']:,} prompt tok   {c['out']:,} out tok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--provider", default="deepseek", choices=list(PROVIDERS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"])
    ap.add_argument("--stateful", action="store_true", help="tried and rejected - see above")
    a = ap.parse_args()
    configure(a.provider, a.model)
    log = ROOT / a.log
    tag = (tag_for(a.provider, CFG["model"]) + ("_V2" if a.prompt == "v2" else "")
           + ("_STATEFUL" if a.stateful else ""))
    out = ROOT / f"data/eval/LLM_JUDGE{tag}_{log.stem}.jsonl"

    rooms = load(str(log))
    done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in out.open(encoding="utf-8") if l.strip()}
    k, sysp = key(), system_prompt(a.stateful, a.prompt == "v2")
    lock, st = threading.Lock(), {"n": 0, "stop": False}
    fh = out.open("a", encoding="utf-8")

    def write(rec):
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st["n"] += 1
            if st["n"] % 250 == 0:
                fh.flush()
                print(f"    {st['n']:,} judged", flush=True)

    def rec(rid, i, msgs, fire, rule, why):
        return {"key": f"{rid}#{i}", "room": rid, "i": i, "sender": msgs[i].get("sender"),
                "text": msgs[i].get("text"), "fire": fire, "rule": rule, "why": why}

    if not a.stateful:
        items = [(rid, i, msgs) for rid, msgs in rooms.items() for i in range(len(msgs))]
        todo = [t for t in items if f"{t[0]}#{t[1]}" not in done]
        if a.limit:
            todo = todo[:a.limit]
        print(f"  {log.name}: {len(rooms)} rooms, {len(items):,} messages   "
              f"{CFG['provider']}/{CFG['model']}   prompt {a.prompt}   "
              f"judged already {len(done):,}   to judge {len(todo):,}")

        def work(t):
            if st["stop"]:
                return
            rid, i, msgs = t
            fire, rule, why = ask(k, sysp, rid, msgs[max(0, i - CTX):i + 1])
            if fire == "AUTH":
                st["stop"] = True
            elif fire != "ERROR":                   # an ERROR is not checkpointed; rerun retries
                write(rec(rid, i, msgs, fire, rule, why))
        units = todo
    else:
        # A room is redone whole unless every message in it is already judged: its state
        # (where sheets opened) has to be rebuilt from the start to be right.
        units = [(rid, msgs) for rid, msgs in rooms.items()
                 if any(f"{rid}#{i}" not in done for i in range(len(msgs)))]
        if a.limit:
            units = units[:a.limit]
        print(f"  {log.name}: {len(rooms)} rooms   stateful   rooms to judge {len(units)}")

        def work(u):
            rid, msgs = u
            shown = {}
            for i in range(len(msgs)):
                if st["stop"]:
                    return
                win = []
                for j in range(max(0, i - CTX), i + 1):
                    win.append(msgs[j])
                    if j in shown and j < i:
                        win.append({"marker": True, "text": shown[j]})
                fire, rule, why = ask(k, sysp, rid, win)
                if fire == "AUTH":
                    st["stop"] = True
                    return
                if fire == "ERROR":
                    fire, rule, why = None, "", "ERROR - treated as quiet"
                if fire:
                    kind = "payment" if fire == "money" else "cab-booking"
                    shown[i] = f"[app: opened the {kind} sheet for {msgs[i].get('sender')} here]"
                write(rec(rid, i, msgs, fire, rule, why))

    with ThreadPoolExecutor(max_workers=1 if a.provider == "ollama" else a.workers) as ex:
        list(ex.map(work, units))
    fh.close()
    if usage_line():
        print("\n" + usage_line())
    if st["stop"]:
        print("  STOPPED: provider rejected the key. Rerun to continue.")
    recs = {}
    for l in out.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            recs[r["key"]] = r
    print(f"  judged {len(recs):,}: {dict(Counter(r['fire'] or 'none' for r in recs.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
