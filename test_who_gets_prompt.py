"""Cases 3, 5 and 7: when the CONFIRMATION comes from the other person, who is the
prompt shown to?

The firing decision reads a 10-message window. detect_target() reads only the current
message's text. On a bare "ok" that leaves it nothing to work with, so this checks
whether show_to lands on the person who actually acts.
"""
import argparse, sys, io, time
import requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CASES = [
  ("3  offer as question -> yeah -> on it", "A", [
      ("A", "should I book a cab?"), ("B", "yeah"), ("A", "on it")]),
  ("7  money offer -> ok -> sending",       "A", [
      ("A", "let me send you 500"), ("B", "ok"), ("A", "sure sending now")]),
  ("8  request -> acceptance (settled: fires)", "A", [
      ("B", "can you book me a cab to the airport"), ("A", "let me book a cab")]),
  ("5  group, C confirms but A asked",      "B/C", [
      ("A", "can someone book me a cab to HSR"), ("B", "let me book a cab"),
      ("C", "okay")]),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8140/detect")
    a = ap.parse_args()
    tag = int(time.time())
    for ci, (name, actor, turns) in enumerate(CASES):
        print(f"\n  {name}   (person who actually acts: {actor})")
        room = f"wgp_{tag}_{ci}"
        for ti, (spk, txt) in enumerate(turns):
            d = requests.post(a.url, timeout=60, json={
                "text": txt, "room_id": room, "sender": spk,
                "message_id": f"{tag}_{ci}_{ti}"}).json()
            got = [i for i in (d.get("intents") or []) if i in ("money", "ride")]
            tgt = (d.get("target") or {})
            tb  = ((d.get("conversation_state") or {}).get("triggered_by") or {})
            if got:
                print(f"      {spk}: {txt!r}")
                print(f"          FIRES {got}   show_to={tgt.get('show_to')!r} "
                      f"({tgt.get('reason')})   triggered_by={tb.get('sender')!r}")
            else:
                print(f"      {spk}: {txt!r}   -")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
