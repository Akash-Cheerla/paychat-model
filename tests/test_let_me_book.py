"""Akash's rule for "let me book a cab", 2026-08-20.

It is never self-sufficient. It is either an ACCEPTANCE (someone already asked) or an
OFFER (needs someone to accept it). Bare, with nothing around it, it must stay quiet.

Compare against "im booking an uber", which FIRING_RULE §3 lists as firing on its own —
present-progressive commitment, not permission-seeking. If the model separates those two
it already understands the rule; if it treats them the same, the rule needs training data.
"""
import argparse, json, sys, io, time
import requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CASES = [
    ("bare self-statement", False, [
        ("1", "hey"), ("2", "yo"), ("1", "let me book a cab")]),
    ("after someone's request", True, [
        ("2", "can you book me a cab from koramangala to the airport"),
        ("1", "let me book a cab")]),
    ("offer, then they accept", True, [
        ("1", "let me book a cab"), ("2", "okay")]),
    ("offer, then they decline", False, [
        ("1", "let me book a cab"), ("2", "nah im good ill walk")]),
    ("§3 control: in-progress, self", True, [
        ("1", "hey"), ("2", "yo"), ("1", "im booking an uber")]),
    ("§3 control: booking FOR them", True, [
        ("2", "im so tired"), ("1", "im booking you a cab right now")]),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8130/detect")
    a = ap.parse_args()
    tag = int(time.time())
    print(f"  {'case':30} {'want':6} {'got':6}  last message")
    bad = 0
    for ci, (name, want, turns) in enumerate(CASES):
        room = f"lmb_{tag}_{ci}"
        fired_on_last = False
        for ti, (spk, txt) in enumerate(turns):
            d = requests.post(a.url, timeout=60, json={
                "text": txt, "room_id": room, "sender": spk,
                "message_id": f"{tag}_{ci}_{ti}"}).json()
            got = [i for i in (d.get("intents") or []) if i in ("money", "ride")]
            sc = (d.get("conversation_state") or {}).get("scores", {})
            if ti == len(turns) - 1:
                fired_on_last = bool(got)
                last_score = sc.get("ride", 0)
        ok = fired_on_last == want
        bad += not ok
        print(f"  {name:30} {str(want):6} {str(fired_on_last):6}  "
              f"ride={last_score:.3f}  {'OK' if ok else '<-- MISMATCH'}")
    print(f"\n  {len(CASES)-bad}/{len(CASES)} match the rule as stated")
    return 1 if bad else 0

if __name__ == "__main__":
    raise SystemExit(main())
