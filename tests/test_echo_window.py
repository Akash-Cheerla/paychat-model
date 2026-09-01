"""Real sequences from the 2026-08-30 log, on both sides of the echo window.

Every one of these scored >=0.99 on the classifier. The question is only whether the
pipeline let it through.

  ECHO      a restatement of the prompt just shown - must stay suppressed
  NEW       a genuinely new commitment later in the room - must fire

dm_13_53 is the case that prompted this: three attempts to send money in a row, none of
which produced anything. The first should fire; the two restatements right after it are
echoes and should not.
"""
import argparse, sys, io, time
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CASES = [
    # ---- must stay suppressed: a restatement 1 message after the prompt ----
    ("ECHO  restated right after the prompt", "money", False, [
        ("A", "hey can you send me 500"),
        ("B", "sure sending it"),          # fires here
        ("B", "sending now"),              # 1 message later - echo
    ]),
    # A second echo case sat here, reconstructed from dm_13_26. Removed: in the real
    # room "let me book a cab too" scored 0.997 and fired, so the following "sure"
    # was an echo of it. In the shortened window it scores 0.105 and does not fire,
    # which makes the "sure" the first prompt for that cab and correct to fire. The
    # case tested a situation that never existed. Real-room behaviour is covered by
    # replay_log_rooms.py, which replays the actual conversation.
    # ---- must fire: a new commitment further down the room ----
    ("NEW   new payment 9 messages later", "money", True, [
        ("A", "can you send me 500"),
        ("B", "sure sending it"),          # fires here
        ("A", "thanks"), ("B", "np"), ("A", "how was your day"),
        ("B", "long one"), ("A", "same"), ("B", "getting food"),
        ("A", "nice"), ("B", "ok"),
        ("B", "sending now"),              # 9 later - a new payment
    ]),
    ("NEW   second cab much later in the room", "ride", True, [
        ("A", "can you book me a cab"),
        ("B", "sure"),                     # fires here
        ("A", "thanks"), ("B", "np"), ("A", "reached"), ("B", "good"),
        ("A", "what time tomorrow"), ("B", "9 ish"), ("A", "ok"), ("B", "cool"),
        ("A", "can you book one for tomorrow too"),
        ("B", "booking now"),              # new request, new commitment
    ]),
    ("NEW   self-initiated, no request in front of it", "money", True, [
        ("A", "can you send me 200"),
        ("B", "sure"),                     # fires here
        ("A", "got it"), ("B", "cool"), ("A", "brb"), ("B", "k"),
        ("A", "back"), ("B", "hows things"), ("A", "fine"), ("B", "same"),
        ("B", "sure i will send it now"),  # the dm_53_69 case, 43 turns in the log
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8900/detect")
    a = ap.parse_args()
    tag = int(time.time())
    bad = 0
    print(f"  {'case':44} {'want':7} {'got':7} {'score':>7}")
    for ci, (name, intent, want, turns) in enumerate(CASES):
        room = f"ew_{tag}_{ci}"
        fired = False
        score = 0.0
        for ti, (spk, txt) in enumerate(turns):
            try:
                d = requests.post(a.url, timeout=60, json={
                    "text": txt, "room_id": room, "sender": spk,
                    "message_id": f"{tag}_{ci}_{ti}"}).json()
            except Exception:
                d = {}
            got = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
            sc = (d.get("conversation_state") or {}).get("scores", {}) or {}
            if ti == len(turns) - 1:
                fired = intent in got
                score = float(sc.get(intent, 0) or 0)
        ok = fired == want
        bad += not ok
        print(f"  {name:44} {'prompt' if want else 'quiet':7} "
              f"{'prompt' if fired else 'quiet':7} {score:7.3f}  {'' if ok else '<-- FAIL'}")
    print(f"\n  {len(CASES)-bad}/{len(CASES)} correct")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
