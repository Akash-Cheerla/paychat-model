"""group_15 from the dogfood log: the second request in the room gets no prompt at all.

    3  11: Can you book cab for me
    4  41: Sure                        <- fired ride, correct
    6  41: Hi Brahma can you book a cab for me from majestic to jp nagar
    7  26: Sure                        <- fired ride, correct
    8  26: can you book a cab for me from majestic to jp nagar 2nd phase
    9  41: Cool                        <- silent
   10  41: Sure                        <- silent
   11  41: Yes                         <- silent
   12  41: Of course                   <- silent
   13  41: Sure                        <- silent

Ruled by Akash on 2026-08-24: turns 9-13 are one agreement repeated by one person, so
exactly one prompt belongs there. Zero is a miss.

Printing the conversation score next to the fire tells us which half is wrong: a high
score with no prompt is the pipeline suppressing it, a low score is the classifier not
seeing it.
"""
import argparse, sys, io, time
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TURNS = [
    ("41", "Hi all"),
    ("41", "Hi"),
    ("11", "Hi"),
    ("11", "Can you book cab for me"),
    ("41", "Sure"),
    ("26", "Hi"),
    ("41", "Hi Brahma can you book a cab for me from majestic to jp nagar"),
    ("26", "Sure"),
    ("26", "can you book a cab for me from majestic to jp nagar 2nd phase"),
    ("41", "Cool"),
    ("41", "Sure"),
    ("41", "Yes"),
    ("41", "Of course"),
    ("41", "Sure"),
    ("41", "Can you ask me again"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8600/detect")
    a = ap.parse_args()
    tag = int(time.time())
    room = f"g15_{tag}"
    print(f"  {'#':>2} {'who':>3}  {'message':46} {'ride':>6}  fired")
    fired_after_8 = 0
    for i, (spk, txt) in enumerate(TURNS):
        d = requests.post(a.url, timeout=60, json={
            "text": txt, "room_id": room, "sender": spk,
            "message_id": f"{tag}_{i}"}).json()
        got = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
        sc = (d.get("conversation_state") or {}).get("scores", {}) or {}
        ride = float(sc.get("ride", 0) or 0)
        if i >= 9 and got:
            fired_after_8 += 1
        flag = ""
        if ride >= 0.825 and not got:
            flag = "   <-- scored high, SUPPRESSED"
        print(f"  {i:2} {spk:>3}  {txt[:46]:46} {ride:6.3f}  {str(got) if got else '-'}{flag}")
    print(f"\n  prompts after the second request (turns 9-13): {fired_after_8}   expected 1")
    return 0 if fired_after_8 == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
