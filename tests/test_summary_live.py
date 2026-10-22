"""Both summarisers, end to end over the HTTP service, on real room traffic.

The unit batteries (test_conversation_summary, test_no_leaks, test_summarizer) run the
engines in-process against fixtures. They cannot catch three things that only show up
live: a leak that survives the serialisation the backend actually receives, a point that
no line in the room ever said, and a recap built from traffic nobody wrote for a test.

So this drives the deployed shape - POST to the running service - over the private
dogfood logs, and asserts the two properties that matter for a feature that renders
other people's chats:

  NO LEAK      no secret, identifier or health detail reaches a card. The detector here
               is written FRESH rather than imported from sensitive.py, because importing
               the thing under test makes the check tautological: it would only prove the
               module agrees with itself.
  GROUNDED     every point is a span of something somebody actually said. The recap path
               is extractive by design, so a point whose content words are absent from
               the window is a bug in selection, not a paraphrase.

The logs are private third-party chats: measurement only, never committed, never training
data. This test reads them if they are present and skips cleanly if they are not, so it
still runs on a fresh clone.

    python tests/test_summary_live.py
    python tests/test_summary_live.py --url http://127.0.0.1:8972 --rooms 12
"""
import argparse
import collections
import glob
import io
import json
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FAILED = []

# An independent leak detector. Deliberately BROADER than sensitive.py and written from
# the threat ("what must never render") rather than from that module's patterns, so the
# two can disagree - which is the only way this check has any power.
LEAK = [
    ("one-time code", re.compile(r"\b(?:otp|one[\s-]?time\s*(?:code|password|pin)|"
                                 r"verification\s*code|auth\s*code)\b", re.I)),
    ("password", re.compile(r"\b(?:password|passcode|passphrase|pwd)\b", re.I)),
    ("card detail", re.compile(r"\b(?:cvv|cvc|card\s*(?:no|num|number)|expiry)\b", re.I)),
    ("bank pin", re.compile(r"\b(?:pin\s*(?:code|number|is)|upi\s*pin|atm\s*pin)\b", re.I)),
    ("api secret", re.compile(r"\b(?:api[_\s-]?key|secret\s*key|access\s*token|"
                              r"auth\s*token|bearer\s+[\w.-]{8,})\b", re.I)),
    ("secret-looking token", re.compile(r"\bsk-[\w-]{8,}\b")),
    ("govt id", re.compile(r"\b(?:aadhaar|aadhar|pan\s*card|ssn|"
                           r"passport\s*(?:no|number))\b", re.I)),
    ("bank id", re.compile(r"\b(?:ifsc|account\s*(?:no|num|number)|routing\s*number)\b", re.I)),
    ("long digit run", re.compile(r"\b\d{10,}\b")),
    ("card-shaped digits", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}(?:[\s-]?\d{4})?\b")),
    ("health detail", re.compile(r"\b(?:biopsy|blood\s*test|mri|x-?ray|chemo|"
                                 r"diagnosis|diagnosed|tumour|tumor)\b", re.I)),
]

STOP = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "they", "we", "me", "him", "her", "them",
    "us", "my", "your", "his", "their", "our", "do", "does", "did", "will", "would", "can",
    "could", "should", "has", "have", "had", "not", "no", "so", "as", "up", "out", "then",
    "there", "here", "what", "when", "who", "how", "all", "any", "one", "now", "just",
}


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in STOP and len(w) > 1]


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def load_rooms(limit):
    """Real room-days from the private logs, busiest first. Empty if they are absent."""
    rows, seen = [], set()
    for p in sorted(glob.glob(str(ROOT / "data" / "eval" / "dogfood_*.jsonl"))):
        for line in io.open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            text = (r.get("text") or "").strip()
            room, ts = r.get("room") or "", r.get("ts") or ""
            if not text or not room or not ts:
                continue
            key = (room, ts, text[:60])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"room": room, "sender": str(r.get("sender") or "?"),
                         "ts": ts, "text": text})
    byday = collections.defaultdict(list)
    for r in rows:
        byday[(r["room"], r["ts"][:10])].append(r)
    ordered = sorted(byday.items(), key=lambda kv: -len(kv[1]))
    return [(k, sorted(v, key=lambda r: r["ts"])) for k, v in ordered[:limit]]


def leaks_in(text):
    return [label for label, rx in LEAK if rx.search(text or "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8972")
    ap.add_argument("--rooms", type=int, default=12)
    ap.add_argument("--messages", type=int, default=40)
    ap.add_argument("--show", action="store_true", help="print every recap")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    s = requests.Session()

    try:
        s.get(f"{base}/health", timeout=20).raise_for_status()
    except Exception as e:
        print(f"  summary service not reachable at {base} - start summary_app first ({e})")
        return 1

    rooms = load_rooms(a.rooms)
    if not rooms:
        print("  no dogfood logs present - nothing to measure, skipping cleanly")
        return 0
    print(f"  {len(rooms)} real room-days, busiest first "
          f"({len(rooms[0][1])} messages down to {len(rooms[-1][1])})\n")

    # ---- conversation summary -------------------------------------------------
    recaps = 0
    points_total = 0
    leaked = []
    ungrounded = []
    mismatched = []
    empty = 0
    for (room, day), msgs in rooms:
        body = {"room_id": room,
                "messages": [{"sender": m["sender"], "ts": m["ts"], "text": m["text"]}
                             for m in msgs]}
        r = s.post(f"{base}/conversation-summary", json=body, timeout=180)
        r.raise_for_status()
        summary = r.json().get("summary")
        if not summary:
            empty += 1
            continue
        recaps += 1
        said = set()
        for m in msgs:
            said.update(words(m["text"]))
        # `sections` and the flat `points` list carry the SAME points - the flat list is
        # kept for a client that does not render sections. Scoring both would double every
        # count, so sections are the source here and the duplication is asserted instead.
        flat = [(sec.get("heading") or "?", p.get("text") or "")
                for sec in summary.get("sections") or []
                for p in sec.get("points") or []]
        in_sections = sorted(t for _, t in flat)
        in_flat = sorted((p.get("text") or "") if isinstance(p, dict) else str(p)
                         for p in summary.get("points") or [])
        if in_sections != in_flat:
            mismatched.append((room, day, len(in_sections), len(in_flat)))
        for section, text in flat:
            points_total += 1
            for label in leaks_in(text):
                leaked.append((room, day, label, text))
            w = words(text)
            if w and sum(1 for x in w if x in said) / len(w) < 0.6:
                ungrounded.append((room, day, section, text))
        if a.show:
            print(f"    {room} {day}  shape={summary.get('shape')}")
            for section, text in flat:
                print(f"      [{section}] {text[:92]}")

    print(f"  conversation summary: {recaps} recaps from {len(rooms)} room-days "
          f"({empty} quiet windows -> no card), {points_total} points total")
    check("no secret or identifier reached a recap", not leaked,
          f"{len(leaked)} leak(s)" if leaked else "")
    check("every recap point is a span of something said", not ungrounded,
          f"{len(ungrounded)} ungrounded" if ungrounded else "")
    check("the flat points list matches the sections", not mismatched,
          f"{len(mismatched)} room(s) disagree" if mismatched else "")
    for room, day, label, text in leaked[:6]:
        print(f"      LEAK [{label}] {room} {day}: {text[:92]}")
    for room, day, section, text in ungrounded[:6]:
        print(f"      UNGROUNDED [{section}] {room} {day}: {text[:92]}")

    # ---- per-message summary --------------------------------------------------
    # The longest real messages: the only kind that needs a card, and the kind most
    # likely to carry a secret somewhere in the middle of it.
    allmsgs = [m for _, msgs in rooms for m in msgs]
    longest = sorted(allmsgs, key=lambda m: -len(m["text"]))[:a.messages]
    batch = [{"message_id": f"live_{i}", "text": m["text"]} for i, m in enumerate(longest)]
    r = s.post(f"{base}/message-summary", json={"messages": batch}, timeout=900)
    r.raise_for_status()
    out = r.json()
    results = out.get("results") or out.get("summaries") or []
    cards = 0
    mleaked = []
    for item, src in zip(results, longest):
        card = item.get("summary") if isinstance(item, dict) else None
        if not card:
            continue
        cards += 1
        parts = [card.get("title") or ""]
        parts += [p if isinstance(p, str) else (p or {}).get("text") or ""
                  for p in (card.get("points") or [])]
        for text in parts:
            for label in leaks_in(text):
                mleaked.append((label, text, src["text"]))

    print(f"\n  per-message summary: {cards} cards from the {len(longest)} longest "
          f"real messages")
    check("no secret or identifier reached a card", not mleaked,
          f"{len(mleaked)} leak(s)" if mleaked else "")
    for label, text, src in mleaked[:6]:
        print(f"      LEAK [{label}] card: {text[:80]}")
        print(f"                     from: {src[:80]}")

    print(f"\n  {'all checks passed' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
