"""What satisfies an open request?

From the dogfood log, 2026-08-21: "Can you send me 10$" / "Hi" fired money at 0.997.
A greeting is not an acceptance. This checks how wide that hole is - one word, or every
short reply that happens to follow a request.

Controls matter as much as the failures here. "ok"/"sure"/"yeah" MUST still fire, and per
Gowtham on 2026-08-20 a thumbs up is an explicit confirm, so it must fire too. Any fix
that silences the greetings has to leave those alone.
"""
import argparse, sys, io, time
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REQUEST = ("26", "Can you send me 10$")

MUST_FIRE = ["ok", "sure", "yeah", "yes", "okay", "k", "of course", "will do",
             "sending now", "👍", "done deal", "yeah sure"]
MUST_STAY_QUIET = ["hi", "hello", "hey", "yo", "good morning", "?", "??", "lol",
                   "haha", "hmm", "what", "why", "who", "wait what", "huh",
                   "no", "nope", "cant sorry", "sorry no", "not now", "thanks",
                   "brb", "one sec", "😂", "..."]


def probe(url, reply, tag, i):
    room = f"na_{tag}_{i}"
    for ti, (spk, txt) in enumerate([("53", "hey"), REQUEST]):
        requests.post(url, timeout=60, json={
            "text": txt, "room_id": room, "sender": spk,
            "message_id": f"{tag}_{i}_{ti}"})
    d = requests.post(url, timeout=60, json={
        "text": reply, "room_id": room, "sender": "53",
        "message_id": f"{tag}_{i}_z"}).json()
    got = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
    sc = (d.get("conversation_state") or {}).get("scores", {}) or {}
    return ("money" in got), float(sc.get("money", 0) or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8500/detect")
    a = ap.parse_args()
    tag = int(time.time())
    wrong = 0
    print(f"  after {REQUEST[1]!r}\n")
    print("  MUST FIRE")
    for i, r in enumerate(MUST_FIRE):
        f, s = probe(a.url, r, tag, i)
        wrong += not f
        print(f"    {'ok  ' if f else 'MISS'}  {s:.3f}  {r!r}")
    print("\n  MUST STAY QUIET")
    for i, r in enumerate(MUST_STAY_QUIET):
        f, s = probe(a.url, r, tag, 100 + i)
        wrong += f
        print(f"    {'BAD ' if f else 'ok  '}  {s:.3f}  {r!r}")
    tot = len(MUST_FIRE) + len(MUST_STAY_QUIET)
    print(f"\n  {tot - wrong}/{tot} correct")
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
