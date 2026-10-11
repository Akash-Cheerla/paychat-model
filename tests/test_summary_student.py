"""The production summary path: rules gate, student writes, rules fall back.

`production_summary()` is the split each half measurably earns. Rules drop 90.5% of pure
chatter and 82 of 151 qualifying dogfood messages. The student writes the card, title
included: on 69 cards of real traffic 7.2% of its titles were weak against the rules'
72.9% ("Bug" 15 times, "Build" 11 times), and the two keep identifiers equally (1/20
each) - the "Build 52" vs "App Build Testing" example that once justified the rules
title was a single case, not a pattern. The student rewrites instead of copying: 5.2%
verbatim against the rules' 60.7%, at half the source length, with zero invented numbers
across 326 bullets and 0 of 69 titles.

Neither does the other's job. The student alone cards every piece of small talk it is
handed - it produced a card for the IOS/Android/Ride-Request message that Akash rejected
on sight. The rules alone reprint sentences.

The cases below are mostly about what happens when things go WRONG, because that is what
decides whether this is safe to deploy: no weights on the box, a model that throws, a
model that returns nothing, a model that repeats itself.

    python tests/test_summary_student.py
    python tests/test_summary_student.py --url http://127.0.0.1:8971/detect
"""
import argparse
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import summary_engines as E          # noqa: E402

BUILD52 = ("Build 52 assigned in live. Please test the entirety of the app. It is marked "
           "as a stable release. Ensure 0 issues in positive flow before we hand it over.")
CHATTER = ("IOS. Android. Ride Request. Reply. IOS Issue. IOS. Android. Ride Request. "
           "Reply. IOS Issue. IOS Android Ride Request Reply. IOS Android Ride Request.")
SHORT = "see you at 5"


class Stub:
    """Stands in for the student so the failure modes are testable without 1GB of weights."""

    def __init__(self, result=None, boom=False):
        self.result, self.boom = result, boom

    def summarize(self, text):
        if self.boom:
            raise RuntimeError("model exploded")
        return self.result


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        want {want!r}\n        got  {got!r}")
    return ok


def summary_endpoint(url):
    """The summary URL to check, or None to skip the endpoint half.

    tests/run_all.py hands every battery the same /detect URL, and since 2026-10-11 the
    summary is a separate service on its own port - rewriting /detect to /message-summary
    hits the intent API, which answers 404. So the endpoint half runs only when
    PAYCHAT_SUMMARY_URL says where that service is, or when --url already points at it.
    """
    import os
    env = os.environ.get("PAYCHAT_SUMMARY_URL")
    if env:
        return env.rstrip("/") + "/message-summary"
    if url and "/detect" not in url and "/classify" not in url:
        return url.rstrip("/") + ("" if url.endswith("/message-summary")
                                  else "/message-summary")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="also exercise /message-summary")
    a = ap.parse_args()
    failed = []

    # ---- the gate decides existence, whatever the student would have said -------------
    out = E.production_summary(CHATTER, Stub({"title": "Issues", "points": ["IOS Issue",
                                                                           "Android Issue"]}))
    failed += [] if check("topic labels: no card even though the student wrote one",
                          out, None) else ["gate chatter"]

    out = E.production_summary(SHORT, Stub({"title": "X", "points": ["a b c"]}))
    failed += [] if check("too short: no card", out, None) else ["gate short"]

    # ---- the student writes the whole card, title included ---------------------------
    # This asserted the opposite until 2026-10-11, on the strength of this very example:
    # the stub's "App Build Testing" loses the build number the rules' "Build 52" keeps.
    # Counted over 69 real cards the example turned out not to generalise - the two keep
    # identifiers equally often (1/20 each) while 72.9% of rules titles were generic or a
    # single bare word against the student's 7.2%. So the student's title ships, and a
    # synthetic case where it reads worse is the price of that.
    out = E.production_summary(BUILD52, Stub({"title": "App Build Testing",
                                              "points": ["Build 52 assigned in live",
                                                         "Test entirety of app"]}))
    ok = out and out["title"] == "App Build Testing" and out["points"] == [
        "Build 52 assigned in live", "Test entirety of app"]
    print(f"  {'PASS' if ok else 'FAIL'}  the student writes the title and the bullets")
    if not ok:
        failed.append("student title")

    # and when the student returns bullets but no title, the rules title is the fallback
    out = E.production_summary(BUILD52, Stub({"points": ["Build 52 assigned in live",
                                                         "Test entirety of app"]}))
    ok = out and out["title"] == "Build 52"
    print(f"  {'PASS' if ok else 'FAIL'}  no student title -> the rules title is used "
          f"({out['title'] if out else None!r})")
    if not ok:
        failed.append("title fallback")
        print(f"        got {out!r}")

    # ---- failure modes: every one must degrade to the rules card, never 500 -----------
    rules_card = E.RulesEngine().summarize(BUILD52)
    for label, stub in [("no weights on the box", None),
                        ("model raises", Stub(boom=True)),
                        ("model returns None", Stub(None)),
                        ("model returns an empty card", Stub({"title": "x", "points": []}))]:
        out = E.production_summary(BUILD52, stub)
        ok = out is not None and out["points"] == rules_card["points"]
        print(f"  {'PASS' if ok else 'FAIL'}  falls back to rules: {label}")
        if not ok:
            failed.append(label)
            print(f"        got {out!r}")

    # ---- output guards ---------------------------------------------------------------
    c = E._parse_flat("T | same thing here | same thing here | a different fact entirely")
    failed += [] if check("exact repeat dropped", c["points"],
                          ["same thing here", "a different fact entirely"]) else ["exact dup"]

    # Verbatim from the dogfood run - the student stuttering under beam search. Kept as
    # the real example rather than a shortened one: trimming it drops the shared words
    # that put it over the threshold, and tuning the threshold to pass an invented case
    # would start merging bullets that are genuinely different.
    c = E._parse_flat('T | Naive: "I\'m looking forward to seeing you" | '
                      'Naive: "I\'m looking forward to to see you" | the build ships friday')
    ok = c and len(c["points"]) == 2
    print(f"  {'PASS' if ok else 'FAIL'}  reworded repeat dropped "
          f"({len(c['points']) if c else 0} left of 3)")
    if not ok:
        failed.append("near dup")

    c = E._parse_flat("T | ok fine | @Priya @Raj | a real bullet with actual content")
    failed += [] if check("two-word and name-only bullets dropped", c["points"],
                          ["a real bullet with actual content"]) else ["guards"]

    failed += [] if check("NONE means no card", E._parse_flat("NONE"), None) else ["none"]
    failed += [] if check("malformed means no card, not a crash",
                          E._parse_flat("no separator at all"), None) else ["malformed"]
    failed += [] if check("every bullet dropped means no card",
                          E._parse_flat("T | ok | no"), None) else ["all dropped"]

    # ---- the real student, if its weights are here -----------------------------------
    student = E.default_student()
    if student is None:
        print("\n  (no student weights on this machine - skipped the live model checks)")
    else:
        t0 = time.time()
        out = E.production_summary(BUILD52, student)
        print(f"\n  live student loaded in {time.time()-t0:.0f}s")
        # the identifier must survive into the title; the wording around it is the
        # student's to choose ("Build 52 Test" is as good as "Build 52")
        ok = out and "Build 52" in out["title"] and len(out["points"]) >= 3
        print(f"  {'PASS' if ok else 'FAIL'}  real student on Build 52 -> "
              f"{out['title'] if out else None!r}, {len(out['points']) if out else 0} bullets")
        if not ok:
            failed.append("live student")
        for p in (out or {}).get("points", []):
            print(f"        - {p}")
        ok = E.production_summary(CHATTER, student) is None
        print(f"  {'PASS' if ok else 'FAIL'}  real student + gate: topic labels make no card")
        if not ok:
            failed.append("live chatter")

    url = summary_endpoint(a.url)
    if a.url and not url:
        print("\n  (no summary service URL - set PAYCHAT_SUMMARY_URL to check the "
              "endpoint; the intent URL does not serve it)")
    if url:
        import requests
        r = requests.post(url, json={"text": BUILD52}, timeout=120).json()
        ok = "Build 52" in ((r.get("summary") or {}).get("title") or "")
        print(f"\n  {'PASS' if ok else 'FAIL'}  endpoint returns the card "
              f"(source={r.get('source')!r})")
        if not ok:
            failed.append("endpoint card")
        r = requests.post(url, json={"text": SHORT}, timeout=120).json()
        ok = r.get("summary") is None
        print(f"  {'PASS' if ok else 'FAIL'}  endpoint returns null for a short message")
        if not ok:
            failed.append("endpoint null")

    print(f"\n  TOTAL {'PASS' if not failed else str(len(failed)) + ' FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
