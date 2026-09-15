"""The LLM judge with the SERVER's state in every call - open requests, and sheets already
opened - instead of making it infer them from raw chat.

On real traffic a reasoning LLM beat v14 (blind holdout 63 vs 56 right), and nearly all of
its remaining errors were state, not judgement: misreading who asked, firing again on a
repeat because it cannot see a prompt already went out, a filter guessing what a repeat
is. The server already knows all of that. This hands it over.

Division of labour, the way it would run in production:
  the LLM      judges the final message: does it FIRE, and is it itself a new REQUEST
  this code    keeps the books with conversation.py's own rules -
                 a request opens a pending per (intent, requester); a newer one from the
                 same person replaces it; it expires after PENDING_MSG_LIMIT (20)
                 intervening messages; a fire consumes the latest open pending from
                 someone else and starts an INTENT_COOLDOWN_MSGS (3) cooldown.
               conversation.py ALSO refuses to store a new pending during that cooldown
               unless the wording matches _EXPLICIT. That guard is not copied: it protects
               against the base model reading acceptances as requests, and here the LLM
               says what a request is. Left in, it dropped 20 real requests across the
               three logs - see the note at the pending-store below.

Requests come from the LLM, not the base model: the base model scores acceptances high as
well ("sure will send" 0.787), so it would open pendings on acceptances.

Why this is not the rejected --stateful mode: that put markers in the chat and left the
LLM to decide what counted as "the same request", and it called the first acceptance of
a re-sent request "already answered" 46 times. Here the rules decide what is open; the
LLM only reads the list.

Rooms run in parallel; messages within a room must run in order, because each state is
built from the decisions before it. Output "fire" is what the app would show (after the
backstop); "fire_llm" is the LLM's own answer.

Real people's messages go to the provider - DeepSeek approved by Akash on 2026-09-10 for
measurement. Nothing here is ever training data.

    python data_gen/llm_judge_state.py --log data/eval/dogfood_2026-08-11.jsonl --model deepseek-reasoner
"""
import argparse, json, re, subprocess, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_gen"))
import llm_judge_log as J                     # noqa: E402  (imports replay_ab, which wraps stdout)
from replay_ab import load                     # noqa: E402

MR = ("money", "ride")
CTX = J.CTX
PENDING_MSG_LIMIT = 20                         # conversation.py
INTENT_COOLDOWN_MSGS = 3                       # conversation.py
# How far back the state lists opened sheets. Was 6, which made the judge treat an old
# sheet as covering a re-sent request that was open and waiting - 16 of 18 misses on the
# holdout said "sheet already opened for this request" while open requests sat in the list.
# The server's own memory here is the cooldown, so use that.
SHEETS_SHOWN = INTENT_COOLDOWN_MSGS = 3
# conversation.py's two regexes: during a cooldown only explicit request language opens a
# new pending. Mirrored, not imported - importing conversation.py pulls in torch.
_EXPLICIT = {
    "money": re.compile(r"\b(?:send|venmo|cashapp|zelle|pay|transfer)\s+(?:me|him|her|them|us)\b"
                        r"|\b(?:you|u)\s+(?:owe|still owe)\s+(?:me|us)\b"
                        r"|\b(?:pay\s+(?:me\s+)?back|send\s+(?:me\s+)?\d)", re.I),
    "ride": re.compile(r"\b(?:book|get|call|order|grab)\s+(?:me\s+)?(?:a\s+)?(?:uber|lyft|cab|taxi|ride)\b", re.I),
}

SYSTEM_STATE = """You decide whether a chat app should open a payment sheet or a cab-booking
sheet at this moment, and whether the final message is itself a new request.

Every conversation comes with APP STATE, computed by the app's server. Treat it as fact:
  OPEN REQUESTS  requests still waiting for an answer, and who made each one. The final
                 message can ACCEPT only an open request made by someone other than its
                 sender. Every request in this list is UNANSWERED - accepting one fires,
                 however many sheets were opened before it.
  SHEETS OPENED  prompts that already went out, each naming the one request it answered.
                 A sheet settles THAT request only. It says nothing about any request in
                 OPEN REQUESTS: those are still waiting. People re-send the same wording
                 for a new request, and that is a new request.
With no open request, the final message can still fire as a self-initiated action in
progress (SS3), or as the acceptance of an OFFER made in the chat (SS3c) - offers are not
listed as requests.

Check these before answering - they are the mistakes judges make most on this task:

  FIRES  SS2   a terse acceptance of an OPEN request: "sure", "ok", "done deal", "im on it",
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
  {"fire": "money" | "ride" | "none", "request": "money" | "ride" | "none",
   "rule": "<section>", "why": "<=12 words>"}
"request" is whether the final message itself asks someone else to send money or book a
ride - a new request or the same one sent again. An offer ("shall i send you 20?") is not
a request."""


# v2, 2026-09-14. On 09-10, 11 of v1's 15 false prompts were a SECOND sheet for a payment or
# booking that already had one: the booker saying "booking the ride now" right after their
# "okay, i'll do it" fired; the payer's "sending now" after the other person accepted their
# offer; "yes please" confirming an offer whose sheet had already opened. v1 could not see
# it: sheets were listed for 3 messages, and by who SENT the firing message, not who acts
# on it - accepting 20's offer opens 20's payment, not the accepter's.
V2_JSON = ('  {"fire": "money" | "ride" | "none", "request": "money" | "ride" | "none",\n'
           '   "acts": "<sender>", "rule": "<section>", "why": "<=12 words>"}')
V2_EXTRA = """

ONE SHEET PER PAYMENT OR BOOKING. SHEETS OPENED names who acts on each sheet (the person
who pays or books). If a sheet already opened for the person who would act on the final
message, for the SAME payment or booking, stay QUIET - a second sheet is a duplicate. That
covers: the actor restating their commitment ("okay i'll do it" then "booking the ride
now"), the actor announcing the payment after the other person accepted their offer, and
the other person confirming an offer whose sheet already opened ("can i send it now?" /
"yes please").
Fire again only when something changed after that sheet:
  - the attempt failed or did not work ("not clickable", "didn't go through", "try again")
  - the actor had backed off ("actually i have to check") and now commits again
  - a different request or offer: another person's request, a new amount, a new route
This never silences an OPEN REQUEST made after the sheet by someone else - those are
always unanswered.

Also return "acts": the sender name of the person who pays or books if this fires - the
sender for an acceptance or a self-initiated action, the offerer when the final message
accepts someone's offer. Use "" when it does not fire."""


def system_prompt(variant="v1", rules_rev=None):
    if rules_rev:
        t = subprocess.run(["git", "show", f"{rules_rev}:FIRING_RULE.md"], cwd=ROOT,
                           capture_output=True, check=True).stdout.decode("utf-8")
    else:
        t = (ROOT / "FIRING_RULE.md").read_text(encoding="utf-8")
    i = t.find("## The one principle")
    head = SYSTEM_STATE
    if variant == "v2":
        old_json = ('  {"fire": "money" | "ride" | "none", "request": "money" | "ride" | "none",\n'
                    '   "rule": "<section>", "why": "<=12 words>"}')
        assert old_json in head
        head = head.replace(old_json, V2_JSON) + V2_EXTRA
    return f"{head}\n\n---\n\nTHE RULES:\n\n{t[i:] if i > 0 else t}\n\n{J.ADDENDUM}"


SHEETS_SHOWN_V2 = 12


def state_block(open_reqs, sheets, i, sender, variant="v1"):
    lines = ["APP STATE:", "  OPEN REQUESTS:"]
    mine = [p for p in open_reqs if p["by"] != sender]
    if mine:
        for p in sorted(mine, key=lambda p: -p["i"]):
            lines.append(f"    - {p['intent']}, requested by {p['by']}, {i - p['i']} messages ago: "
                         f"{p['text'][:120]!r}")
    else:
        lines.append("    none")
    window = SHEETS_SHOWN_V2 if variant == "v2" else SHEETS_SHOWN
    lines.append(f"  SHEETS OPENED (last {window} messages):")
    recent = [s for s in sheets if i - s["i"] <= window]
    if recent:
        for s in sorted(recent, key=lambda s: -s["i"]):
            ans = f", answering {s['answering']}'s request" if s.get("answering") else ""
            if variant == "v2":
                what = (f"answering {s['answering']}'s request" if s.get("answering")
                        else "no open request - an offer or self-initiated")
                lines.append(f"    - {s['intent']} sheet, {i - s['i']} messages ago, acted on by "
                             f"{s.get('acts') or s['for']} ({what}); fired on {s['for']}'s "
                             f"{str(s.get('text', ''))[:70]!r}")
            else:
                lines.append(f"    - {s['intent']} sheet opened for {s['for']}, {i - s['i']} messages ago{ans}")
    else:
        lines.append("    none")
    return "\n".join(lines)


def ask(k, sysp, room, window, state, retries=4, acts_box=None):
    kind = "a GROUP chat" if room.startswith("group") else "a one-to-one DM"
    convo = "\n".join(f"{m.get('sender')}: {m.get('text')}" for m in window)
    last = window[-1]
    user = (f"Room type: {kind}.\n\n{state}\n\nTHE CONVERSATION:\n\n{convo}\n\n"
            f"Judge ONLY the final line, sent by {last.get('sender')}: {last.get('text')!r}")
    body = {"model": J.CFG["model"], "temperature": 0,
            "messages": [{"role": "system", "content": sysp}, {"role": "user", "content": user}]}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {k}"}
    for a in range(retries):
        try:
            r = requests.post(J.CFG["url"], timeout=180, headers=headers, json=body)
            if r.status_code in (401, 402, 403):
                return "AUTH", None, "", ""
            if r.status_code != 200:
                time.sleep(min(2 ** a, 30)); continue
            j = r.json()
            u = j.get("usage") or {}
            with J._ULOCK:
                J.USAGE["hit"] += int(u.get("prompt_cache_hit_tokens") or 0)
                J.USAGE["miss"] += int(u.get("prompt_cache_miss_tokens") or 0)
                J.USAGE["out"] += int(u.get("completion_tokens") or 0)
                J.USAGE["calls"] += 1
            d = J.parse(j["choices"][0]["message"].get("content"))
            f = str(d.get("fire", "none")).lower()
            q = str(d.get("request", "none")).lower()
            if acts_box is not None:
                acts_box[0] = str(d.get("acts") or "").strip()
            return (f if f in MR else None), (q if q in MR else None), \
                str(d.get("rule", ""))[:10], str(d.get("why", ""))[:70]
        except Exception:
            time.sleep(min(2 ** a, 30))
    return "ERROR", None, "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--model", default="deepseek-reasoner")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--rooms", type=int, default=0, help="first N rooms only - a trial")
    ap.add_argument("--show-state", type=int, default=0, help="print N state blocks, to check")
    ap.add_argument("--variant", default="v1", choices=["v1", "v2"],
                    help="v2: who acts on each sheet, 12-message sheet list, one sheet per payment")
    ap.add_argument("--rules-rev", default=None,
                    help="read FIRING_RULE.md from this git revision, to hold the rules text fixed")
    ap.add_argument("--sheets-window", type=int, default=None,
                    help="v2: how many messages back SHEETS OPENED lists (default 12)")
    ap.add_argument("--run", default="",
                    help="label for a repeat run, so it writes its own file (e.g. R2)")
    ap.add_argument("--backstop", action="store_true",
                    help="also apply the server's cooldown rule on top - off by default, see above")
    a = ap.parse_args()
    J.configure("deepseek", a.model)
    log = ROOT / a.log
    global SHEETS_SHOWN_V2
    if a.sheets_window:
        SHEETS_SHOWN_V2 = a.sheets_window
    vtag = ("" if a.variant == "v1" else a.variant.upper()) + \
        (f"W{a.sheets_window}" if a.sheets_window else "") + a.run
    out = ROOT / f"data/eval/LLM_JUDGE_STATE{vtag}-{re.sub(r'[^A-Za-z0-9]+', '-', a.model)}_{log.stem}.jsonl"

    rooms = load(str(log))
    done = {}
    if out.exists():
        for l in out.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["key"]] = r
    units = [(rid, msgs) for rid, msgs in rooms.items()
             if any(f"{rid}#{i}" not in done for i in range(len(msgs)))]
    if a.rooms:
        units = units[:a.rooms]
    print(f"  {log.name}: {len(rooms)} rooms, {sum(len(m) for m in rooms.values()):,} messages   "
          f"deepseek/{a.model} + server state   rooms to judge {len(units)}", flush=True)

    k, sysp = J.key(), system_prompt(a.variant, a.rules_rev)
    lock, st = threading.Lock(), {"n": 0, "stop": False, "shown": 0}
    fh = out.open("a", encoding="utf-8")

    def work(u):
        rid, msgs = u
        pend, sheets, cool = [], [], {}
        for i, m in enumerate(msgs):
            if st["stop"]:
                return
            s = str(m.get("sender"))
            pend = [p for p in pend if i - p["i"] <= PENDING_MSG_LIMIT]       # expiry
            state = state_block(pend, sheets, i, s, a.variant)
            box = [""]
            fire, req, rule, why = ask(k, sysp, rid, msgs[max(0, i - CTX):i + 1], state,
                                       acts_box=box)
            if fire == "AUTH":
                st["stop"] = True
                return
            if fire == "ERROR":
                fire, req, why = None, None, "ERROR - treated as quiet"
            # The server's backstop: no second sheet for the same sender and intent inside
            # the cooldown, unless a request from someone else is open for that intent.
            # OFF by default: on the holdout it fired twice and was wrong twice, dropping
            # self-initiated retries after a failed payment ("Sending you ₹100 again",
            # "Sending you 20 rupees"). The state block now tells the judge what already
            # opened, so the blunt rule on top only removes real prompts.
            shown = fire
            if a.backstop and fire and cool.get(fire, 0) > 0 and any(
                    sh["intent"] == fire and sh["for"] == s and i - sh["i"] <= INTENT_COOLDOWN_MSGS
                    for sh in sheets) and not any(
                    p["intent"] == fire and p["by"] != s for p in pend):
                shown = None
            for x in list(cool):
                cool[x] -= 1
                if cool[x] <= 0:
                    del cool[x]
            if shown:                                    # consume the latest open request
                cands = [p for p in pend if p["intent"] == shown and p["by"] != s]
                ans = None
                if cands:
                    p = max(cands, key=lambda p: p["i"])
                    ans = p["by"]
                    pend.remove(p)
                sheets.append({"intent": shown, "for": s, "i": i, "answering": ans,
                               "acts": box[0] or s, "text": m.get("text") or ""})
                cool[shown] = INTENT_COOLDOWN_MSGS
            if req and req != shown:                     # open (or replace) a pending
                # conversation.py also suppresses a new pending during the cooldown unless
                # the wording matches _EXPLICIT. That guard exists because the BASE model
                # mis-reads acceptances as requests; here the LLM says what is a request,
                # so the guard only loses real ones. On 2026-09-11 it dropped Gowtham's
                # "Can you book from my home to office" (no ride noun) three messages after
                # a ride fired, so "Lemme book it" had nothing to accept and stayed quiet -
                # the miss Akash reported as "I didn't get any intent". The live app misses
                # it too: the same rule runs in production.
                pend = [p for p in pend if not (p["intent"] == req and p["by"] == s)]
                pend.append({"intent": req, "by": s, "text": m.get("text") or "", "i": i})
            rec = {"key": f"{rid}#{i}", "room": rid, "i": i, "sender": s, "text": m.get("text"),
                   "fire": shown, "fire_llm": fire, "request": req, "rule": rule, "why": why,
                   "acts": box[0],
                   "open": len([p for p in pend if p["by"] != s])}
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                st["n"] += 1
                if st["shown"] < a.show_state and (fire or req):
                    st["shown"] += 1
                    print(f"\n  --- [{rid}#{i}] {s}: {str(m.get('text'))[:60]!r}  ->  fire={fire} "
                          f"shown={shown} request={req}  ({rule}: {why})\n{state}", flush=True)
                if st["n"] % 250 == 0:
                    fh.flush()
                    print(f"    {st['n']:,} judged", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, units))
    fh.close()
    line = J.usage_line()
    if line:
        print("\n" + line)
    if st["stop"]:
        print("  STOPPED: provider rejected the key. Rerun to continue.")
    recs = {}
    for l in out.open(encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            recs[r["key"]] = r
    backstop = sum(1 for r in recs.values() if r.get("fire_llm") and not r.get("fire"))
    print(f"  judged {len(recs):,}: shown {dict(Counter(r['fire'] or 'none' for r in recs.values()))}"
          f"   dropped by the backstop {backstop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
