"""Where self-initiated actually stands, on the model rather than the pipeline.

Two separate things were failing and only one is fixed:

  pipeline   a self-initiated commitment after an earlier fire was suppressed for four
             hours unless it named a new amount. Fixed - three recovered on the real log.
  model      whether the classifier recognises the commitment at all. Untouched.

This is the second question. Every line is self-initiated: the speaker says they will do
it, or are doing it, with nobody having asked. Phrasings are taken from the real logs and
the screenshots, not invented, and each is sent into an empty room so nothing else can
carry it.

Per FIRING_RULE 3 and 3c: an action in progress fires, a need never does, a future
promise never does, and an unprompted offer waits for the other person.
"""
import argparse, sys, io, time
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# (message, intent, should_fire, where it came from)
CASES = [
    # --- in progress: must fire ---
    ("sending now",                                   "money", True,  "dm_13_53"),
    ("Sure i will send it now",                       "money", True,  "dm_53_69"),
    ("100$ paying you now",                           "money", True,  "dm_13_14"),
    ("I'll transfer it now",                          "money", True,  "screenshot"),
    ("I am booking the cab now",                      "ride",  True,  "dm_13_51"),
    ("im booking an uber",                            "ride",  True,  "spec 3"),
    ("booking now",                                   "ride",  True,  "eval"),
    ("Got it,i will transfer 2500 hundred to you.",   "money", True,  "dm_53_69"),
    ("I'll be sending you that dollar that I owe you", "money", True,  "dm_10_4"),
    ("Okay I'll get you a cab for it then",           "ride",  True,  "screenshot"),
    ("i'll send it",                                  "money", True,  "Anupam"),
    ("Sure,i will book it for you.",                  "ride",  True,  "dm_53_69"),
    ("transferring the money now",                    "money", True,  "-"),
    ("ok booking your ola",                           "ride",  True,  "-"),
    # --- must NOT fire ---
    ("let me book a cab",                             "ride",  False, "spec 3c, offer"),
    ("let me send you 500",                           "money", False, "spec 3c, offer"),
    ("I'll need to book a cab ride back home",        "ride",  False, "spec 3c, need"),
    ("I'll transfer you tonight",                     "money", False, "screenshot, future"),
    ("i'll pay on the 1st",                           "money", False, "Anupam, future"),
    ("just sent it",                                  "money", False, "spec 3a, done"),
    ("cab booked",                                    "ride",  False, "spec 3a, done"),
    ("one sec opening the app",                       "money", False, "spec 3b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8900/detect")
    a = ap.parse_args()
    tag = int(time.time())
    bad = 0
    print(f"  {'message':46} {'want':6} {'got':6} {'score':>7}  source")
    for i, (txt, intent, want, src) in enumerate(CASES):
        try:
            d = requests.post(a.url, timeout=60, json={
                "text": txt, "room_id": f"si_{tag}_{i}", "sender": "A",
                "message_id": f"{tag}_{i}"}).json()
        except Exception:
            d = {}
        got = intent in [x for x in (d.get("intents") or []) if x in ("money", "ride")]
        sc = (d.get("conversation_state") or {}).get("scores", {}) or {}
        s = float(sc.get(intent, 0) or 0)
        ok = got == want
        bad += not ok
        print(f"  {txt[:46]:46} {'fire' if want else 'quiet':6} "
              f"{'fire' if got else 'quiet':6} {s:7.3f}  {src}{'' if ok else '   <-- FAIL'}")
    print(f"\n  {len(CASES)-bad}/{len(CASES)} correct")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
