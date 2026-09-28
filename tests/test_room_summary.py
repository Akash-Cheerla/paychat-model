"""/room-summary/{room_id} — the live-traffic view of what is open and what was prompted.

Built for the backend on 2026-09-27, because `/summary/{room_id}/{user_name}` only ever saw
the demo WebSocket chat and returned nothing for a real room.

Every check drives the room through /detect first, so this tests what the endpoint reports
about traffic the classifier actually decided - not a fixture.

    python tests/test_room_summary.py --url http://127.0.0.1:8971/detect
"""
import argparse
import io
import json
import sys
import time

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8900/detect")
    a = ap.parse_args()
    base = a.url.rsplit("/", 1)[0]
    tag = int(time.time())
    s = requests.Session()

    def send(room, sender, text, i, participants=None):
        body = {"text": text, "room_id": room, "sender": str(sender),
                "message_id": f"{room}_{i}"}
        if participants:
            body["participants"] = participants
        return s.post(a.url, json=body, timeout=90).json()

    def summary(room, **q):
        r = s.get(f"{base}/room-summary/{room}", params=q, timeout=60)
        return r.status_code, r.json()

    # 1. a DM: request, then the acceptance that fires
    room = f"dm_rs{tag}_1"
    send(room, 1, "can you send me 500 for lunch", 0)
    d = send(room, 2, "sure", 1)
    fired = [x for x in (d.get("intents") or []) if x in ("money", "ride")]
    code, out = summary(room)
    check("endpoint answers 200", code == 200, f"got {code}")
    check("the request that was answered is no longer open",
          not any(o["intent"] == "money" for o in out.get("open_requests", [])),
          json.dumps([o["text"][:30] for o in out.get("open_requests", [])]))
    p = [x for x in out.get("prompts", []) if x["intent"] == "money"]
    check("the prompt is listed", bool(p) == bool(fired), f"prompts={len(p)} fired={fired}")
    if p:
        check("it goes to the payer", p[-1]["to"] == "2", f"to={p[-1]['to']}")
        check("it carries the request's amount", p[-1]["slots"].get("amount") == "$500",
              json.dumps(p[-1]["slots"]))
        check("it names the request it answers",
              (p[-1].get("answers") or {}).get("message_id") == f"{room}_0",
              json.dumps(p[-1].get("answers")))
    check("messages_seen counts every message", out.get("messages_seen") == 2,
          str(out.get("messages_seen")))

    # 2. an unanswered request stays open, with its slots
    room = f"dm_rs{tag}_2"
    send(room, 1, "can you book me a cab to the airport", 0)
    _, out = summary(room)
    opens = out.get("open_requests", [])
    check("an unanswered request is open", len(opens) == 1 and opens[0]["intent"] == "ride",
          json.dumps([(o["intent"], o["text"][:24]) for o in opens]))
    if opens:
        check("the open request carries its slots", opens[0]["slots"].get("destination") == "Airport",
              json.dumps(opens[0]["slots"]))
        check("the open request says who asked", opens[0]["from"] == "1", opens[0]["from"])

    # 3. a group split stays open for the people who have not paid, and both prompts are listed
    room = f"group_rs{tag}_3"
    send(room, 1, "dinner was 900, everyone send me 300", 0, participants=3)
    send(room, 2, "sending now", 1, participants=3)
    send(room, 3, "sending mine too", 2, participants=3)
    _, out = summary(room)
    tos = [x["to"] for x in out.get("prompts", []) if x["intent"] == "money"]
    check("both payers on a split get a prompt", tos == ["2", "3"], json.dumps(tos))
    split = [o for o in out.get("open_requests", []) if o["divisible"]]
    check("the split is still open", bool(split), json.dumps([o["text"][:30] for o in split]))
    if split:
        check("it records who has paid", sorted(split[0]["answered_by"]) == ["2", "3"],
              json.dumps(split[0]["answered_by"]))

    # 4. ?sender= answers "what do I have to act on"
    _, out = summary(room, sender="2")
    check("sender filter keeps only that person's prompts",
          all(x["to"] == "2" for x in out.get("prompts", [])),
          json.dumps([x["to"] for x in out.get("prompts", [])]))
    check("open_requests holds only what OTHERS asked",
          all(o["from"] != "2" for o in out.get("open_requests", [])),
          json.dumps([o["from"] for o in out.get("open_requests", [])]))
    check("it says who it is for", out.get("for") == "2", str(out.get("for")))

    # my own asks are sorted out, not hidden
    room = f"dm_rs{tag}_4"
    send(room, 1, "can you send me 500 for lunch", 0)          # asked by 1
    send(room, 2, "can you book me a cab to the airport", 1)   # asked by 2
    _, out = summary(room, sender="2")
    theirs = [o["from"] for o in out.get("open_requests", [])]
    mine = [o["from"] for o in out.get("my_requests", [])]
    check("what is waiting on me lists the other person's ask", theirs == ["1"], json.dumps(theirs))
    check("my own ask is in my_requests", mine == ["2"], json.dumps(mine))
    _, out = summary(room)
    check("without sender, both are in open_requests",
          sorted(o["from"] for o in out.get("open_requests", [])) == ["1", "2"],
          json.dumps([o["from"] for o in out.get("open_requests", [])]))
    check("without sender there is no my_requests", "my_requests" not in out)

    # a fresh request is not expired; the flag is always present so the app can rely on it
    fresh = (out.get("open_requests") or [{}])[0]
    check("expired is reported and false for a fresh request", fresh.get("expired") is False,
          json.dumps({k: fresh.get(k) for k in ("age_seconds", "expired")}))

    # 5. a room nobody has messaged
    code, out = summary(f"dm_rs{tag}_never")
    check("an unknown room answers empty, not an error",
          code == 200 and out.get("open_requests") == [] and out.get("prompts") == [],
          json.dumps(out)[:120])

    print(f"\n  {len(FAILED)} failed" if FAILED else "\n  all checks passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
