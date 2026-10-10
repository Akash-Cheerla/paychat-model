"""The Summary tab: does the card say something worth scrolling back for?

Akash, 2026-10-07, with two screenshots. The second was right; the first contained this,
and it is the case this file exists for:

    IOS Issue
      - IOS
      - Android
      - Ride Request
      - Reply
      - IOS Issue

Five topic labels and the heading repeated. Nothing a reader could act on, and a card that
costs them the scroll. The purpose of the tab is the opposite: "summary of actions or
something crucial that people talk about, so users can later check".

These run against the module directly, with no server, so they are quick enough to run on
every change. The last block goes through /message-summary to check the endpoint agrees.

    python tests/test_summarizer.py
    python tests/test_summarizer.py --url http://127.0.0.1:8971/message-summary
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from summarizer import summarize, is_worth_summarizing   # noqa: E402

FIGMA = "https://www.figma.com/design/DVFotKidOe78MnTRjCFGKp/SEE?node-id=2991-11531"
FIGMA2 = "https://www.figma.com/design/DVFotKidOe78MnTRjCFGKp/SEE?node-id=2869-11477"

# (name, message, expected title or None for "no card", substrings every point set must contain)
CASES = [
    # ---- the cards from the screenshot that was right -----------------------------
    ("build 52",
     "Build 52 assigned in live. Please test the entirety of the app. It is marked as a "
     "stable release. Ensure 0 issues in positive flow before we hand it over.",
     "Build 52",
     ["Build 52 assigned in live", "Test the entirety of the app", "stable release"]),

    ("overlap issues",
     "End call when top banner disappears. White space remains on all modules. Ideal "
     "spacing should revert back to original once banner disappears. This is creating "
     "overlap issues on certain modules.",
     "Layout",
     ["White space remains on all modules", "overlap issues on certain modules"]),

    ("voice note and calls",
     "The voice note saves in 15-20 seconds which is far too slow for a normal message. "
     "It also auto-disconnects calls after 2-3 attempts, every single time.",
     "Call",
     ["15-20 seconds", "2-3 attempts"]),

    ("verify build",
     "Verify Ride Intents in the new build. Verify Locations under Settings as well. "
     "Verify the fixed bugs from last week before we sign this off.",
     "Testing",
     ["Verify Ride Intents", "Verify Locations under Settings"]),

    ("security and releases",
     "Please check your mail for the Fri Lite security issues that came in overnight. "
     "Also check for the next releases and let me know which ones are blocking.",
     "Security",
     ["Check your mail for the Fri Lite security issues"]),

    # A reference list IS its links. Shortening them to "figma.com link" would leave a
    # card nobody can tap - and this was one of the cards called right.
    ("design reference keeps its links",
     f"Restoring a chat backup after login: {FIGMA}\nDeleting a location: {FIGMA2}",
     "Design Reference",
     [FIGMA, FIGMA2, "Restoring a chat backup after login"]),

    # ---- the card that should never have existed ----------------------------------
    ("topic labels produce no card",
     "IOS. Android. Ride Request. Reply. IOS Issue. IOS. Android. Ride Request. Reply. "
     "IOS Issue. IOS Android Ride Request Reply. IOS Android Ride Request.",
     None, []),

    ("a short message is not summarised", "see you at 5", None, []),
    ("one long sentence is not summarised",
     "I was thinking that maybe sometime next week we could possibly get together and "
     "talk about the thing we discussed the other day if that works for you at all",
     None, []),
    ("a bare link is not summarised", FIGMA, None, []),
    ("two bare links are not summarised", f"{FIGMA}\n{FIGMA2}", None, []),
]

# (name, message, substrings that must NOT appear anywhere in the card)
NEGATIVES = [
    ("pleasantries are dropped",
     "Hi team, hope you are all doing well after the long weekend. Please upload the "
     "signed contract before Friday. Make sure to include the annexes we discussed. "
     "Thanks so much and have a great weekend everyone!",
     ["Hi team", "have a great weekend", "hope you are all"]),
    ("the title is never repeated as a bullet",
     "Build 52 assigned in live. Build 52. Please test the entirety of the app and make "
     "sure the positive flow has zero issues before this goes out to anyone.",
     []),   # checked structurally below
    ("a URL never appears inside a prose bullet",
     f"Please review the latest spacing changes in the file at {FIGMA} before standup. "
     f"Let me know whether the banner behaviour matches what we agreed in the call "
     f"yesterday, because the spacing looked wrong on two of the modules.",
     ["figma.com/design"]),
]


def run_unit():
    failed = []
    for name, msg, want_title, must_contain in CASES:
        d = summarize(msg)
        if want_title is None:
            ok = d is None
            got = "no card" if d is None else f"{d.title}: {d.points}"
        else:
            ok = d is not None and d.title == want_title and all(
                any(s.lower() in p.lower() for p in d.points) for s in must_contain)
            got = "no card" if d is None else f"{d.title}: {d.points}"
        if not ok:
            failed.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"        want {want_title!r} + {must_contain}\n        got  {got}")

    for name, msg, banned in NEGATIVES:
        d = summarize(msg)
        blob = "" if d is None else (d.title + " " + " ".join(d.points)).lower()
        ok = d is not None and all(b.lower() not in blob for b in banned)
        # "the title is never repeated as a bullet"
        if d is not None:
            ok = ok and all(p.lower() != d.title.lower() for p in d.points)
        if not ok:
            failed.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"        got {None if d is None else (d.title, d.points)}")

    total = len(CASES) + len(NEGATIVES)
    print(f"\n  TOTAL {total - len(failed)}/{total} correct")
    return failed


def run_endpoint(url):
    import requests
    print(f"\n  through {url}")
    failed = []
    r = requests.post(url, json={"text": CASES[0][1]}, timeout=30).json()
    one = (r.get("summary") or {}).get("title")
    ok = one == "Build 52"
    failed += [] if ok else ["endpoint: single message"]
    print(f"  {'PASS' if ok else 'FAIL'}  endpoint returns a card for a long message ({one!r})")

    r = requests.post(url, json={"text": "see you at 5"}, timeout=30).json()
    ok = r.get("summary") is None
    failed += [] if ok else ["endpoint: short message"]
    print(f"  {'PASS' if ok else 'FAIL'}  endpoint returns null for a short message")

    batch = [{"message_id": f"m{i}", "text": c[1]} for i, c in enumerate(CASES[:4])]
    r = requests.post(url, json={"messages": batch}, timeout=60).json()
    res = r.get("results") or []
    ok = len(res) == 4 and all(x.get("message_id") == f"m{i}" for i, x in enumerate(res))
    failed += [] if ok else ["endpoint: batch"]
    print(f"  {'PASS' if ok else 'FAIL'}  batch keeps message_id and order ({len(res)} back)")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="also check the endpoint, e.g. .../message-summary")
    a = ap.parse_args()
    failed = run_unit()
    if a.url:
        # tests/run_all.py hands every battery the same /detect URL. Point it at the
        # endpoint this file is about rather than failing on a response it never asked for.
        url = a.url.replace("/detect", "/message-summary")
        if not url.endswith("/message-summary"):
            url = url.rstrip("/") + "/message-summary"
        failed += run_endpoint(url)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
