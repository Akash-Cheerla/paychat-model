"""How many moments needed a prompt, and how many got one — on real chats, in English.

Every chat is translated first so language is not a confound: in Telugu the model cannot
read the message at all, and "missed" would mean nothing.

"Needed a prompt" is decided by FIRING_RULE, applied to the message and what came before:

  fires    someone agrees to a request that is still open ("ok", "sure", "i'll do it")
           someone states the action in progress ("sending now", "i'll send it", "on it")
  does not a request nobody has answered - the asker is not the one acting
           a future promise ("i'll send it next month", "by friday")
           already done ("just sent it", "paid it yesterday")
           talking about cost, splitting a plan, or who owes what in the abstract

Candidates are found by pattern and then printed in full, because a regex over
translated chat will get some wrong and the list is the only way to check. The counts
are only worth what the list is worth.
"""
import argparse, json, re, sys, io, time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.stdout = io.TextIOWrapper(open(sys.__stdout__.fileno(), "wb", closefd=False),
                              encoding="utf-8", errors="replace", line_buffering=True)
MR = ("money", "ride")

# someone is asking the other person to pay or to book
REQUEST = re.compile(
    r"\b(?:can|could|will|would|pls|please)\b.{0,28}\b(?:send|pay|transfer|gpay|venmo|"
    r"zelle|book|cab|uber|ola|ride|drop|pick)\b"
    r"|\b(?:send|pay|transfer|gpay|venmo|zelle|spot|lend)\s+(?:me|us)\b"
    r"|\bbook\s+(?:me\s+)?(?:a\s+)?(?:cab|uber|ola|ride|taxi)\b"
    r"|\btell me the amount\b|\bhow much do i owe\b|\bpending\b", re.I)

# a commitment: agreeing, or saying it is happening now
AGREE = re.compile(r"^(?:ok|okay|k|kk|sure|yeah|yea|ya|yep|yup|yes|fine|of course|"
                   r"why not|will do|done deal|sure thing|👍)\b", re.I)
DOING = re.compile(r"\b(?:i'?ll send|i will send|sending|i'?ll pay|paying it|"
                   r"i'?ll transfer|transferring|i'?ll book|booking|i'?ll do it|"
                   r"on it|doing it|i'?ll get you|i got you|let me send|let me book)\b", re.I)

# things that look like commitments but must not fire
FUTURE = re.compile(r"\b(tomorrow|next week|next month|by (?:mid|early|friday|monday|"
                    r"sunday|then)|later|after (?:i|we) get|once i get|when i get|"
                    r"end of the month|salary)\b", re.I)
DONE   = re.compile(r"\b(just sent|already sent|already paid|sent it|paid it|"
                    r"transferred it|i sent|have sent|booked it already)\b", re.I)


def load(p):
    rows = json.loads(Path(p).read_text(encoding="utf-8"))
    return [(r["who"], r["text"], r.get("original", "")) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8800/detect")
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--show", type=int, default=100)
    a = ap.parse_args()

    CHATS = [("Rakesh", "data/eval/REAL_rakesh_en.json"),
             ("Anupam", "data/eval/REAL_anupam_en.json"),
             ("Everts (roommates)", "data/eval/REAL_everts_en.json")]

    grand = [0, 0, 0, 0]     # needed, fired-on-needed, total fires, messages
    for name, src in CHATS:
        msgs = load(ROOT / src)
        tag = int(time.time())
        fired_at = set()
        for i, (who, text, _) in enumerate(msgs):
            try:
                d = requests.post(a.url, timeout=40, json={
                    "text": text, "room_id": f"cr_{tag}_{i//a.chunk}",
                    "sender": str(who), "message_id": f"{tag}_{i}"}).json()
            except Exception:
                continue
            if [x for x in (d.get("intents") or []) if x in MR]:
                fired_at.add(i)

        # a request is "open" if asked by someone else within the last 8 messages
        needed = []
        for i, (who, text, orig) in enumerate(msgs):
            if FUTURE.search(text) or DONE.search(text):
                continue
            open_req = any(REQUEST.search(msgs[j][1]) and msgs[j][0] != who
                           for j in range(max(0, i - 8), i))
            is_commit = DOING.search(text) or (AGREE.match(text.strip()) and open_req)
            if is_commit and (open_req or DOING.search(text)):
                needed.append(i)

        hit = [i for i in needed if i in fired_at]
        miss = [i for i in needed if i not in fired_at]
        extra = sorted(fired_at - set(needed))
        print(f"\n  ===== {name} =====")
        print(f"    messages                {len(msgs)}")
        print(f"    needed a prompt         {len(needed)}")
        print(f"    of those, fired         {len(hit)}")
        print(f"    of those, MISSED        {len(miss)}")
        print(f"    fired elsewhere         {len(extra)}  (check these are not false)")
        if miss:
            print(f"\n    MISSED:")
            for i in miss[:a.show]:
                who, text, orig = msgs[i]
                print(f"      {who}: {text[:66]}")
                if orig and orig.strip().lower() != text.strip().lower():
                    print(f"         (was: {orig[:60]})")
        if hit:
            print(f"\n    FIRED correctly:")
            for i in hit[:a.show]:
                print(f"      {msgs[i][0]}: {msgs[i][1][:66]}")
        if extra:
            print(f"\n    FIRED elsewhere:")
            for i in extra[:a.show]:
                prev = msgs[i-1][1][:38] if i else ""
                print(f"      after {prev!r}")
                print(f"        {msgs[i][0]}: {msgs[i][1][:60]}")
        grand[0] += len(needed); grand[1] += len(hit)
        grand[2] += len(fired_at); grand[3] += len(msgs)

    print(f"\n  ===== ALL THREE =====")
    print(f"    messages          {grand[3]}")
    print(f"    needed a prompt   {grand[0]}")
    print(f"    fired             {grand[1]}  = {grand[1]/max(grand[0],1)*100:.0f}%")
    print(f"    missed            {grand[0]-grand[1]}")
    print(f"    total prompts     {grand[2]}  ({grand[2]/max(grand[3],1)*1000:.1f} per 1000 messages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
