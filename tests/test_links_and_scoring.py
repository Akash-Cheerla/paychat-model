"""Links are content, and a bug report says "cannot".

Both of these came out of the 2026-10-10 dogfood log, which was the first real traffic
this feature had seen that nothing had been tuned on. Seven of fourteen long messages
produced no card, and two of the cards that did appear were titled after whoever the
message was addressed to.

    Akash, 2026-10-11: "urls should be taken in summary"

What the log showed, and what each case here holds:

  a build or design link IS the message       "please use the following build <url>"
                                              measured 124 chars with the URL deleted and
                                              was dropped for being too short
  a bug list says "cannot"                    "1. Cannot set group profiles pictures" scored
                                              zero - "unable to" matched, a bare "cannot"
                                              did not
  plurals                                     the pattern read "issue\\b", which cannot match
                                              "issues", the commonest bug word in this chat

The negative cases matter more than the positive ones. Every one of these rules can buy a
card for chatter, and two of them did before being tightened: "I can't stop thinking about
this blog post" and a described link to example.com each cost chatter rejection outright.

    python tests/test_links_and_scoring.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import summarizer as S                                      # noqa: E402
from summary_engines import _restore_links  # noqa: E402

BUILD = ("Hi Team, please use the following build for testing. "
         "https://install.shareipa.com/ccUH8I 1. The group profile image issues and UI "
         "issues have been addressed.")
BUGS = ("Group Profile Pic I can see now Brahma. Not sure if these are addressed. "
        "1. Cannot set group profiles pictures once created 2. Cannot make any other "
        "user as admin")
# As WhatsApp actually sends it: the line break after the link is what ends the sentence.
# Written as one run-on line this does NOT card, because is_worth_summarizing still wants
# two sentences - see ONE_LINE below, which pins that boundary deliberately.
FIGMA = ("Hi team, here's the link to the complete user flow for a group call:\n"
         "https://www.figma.com/design/DVFotKid0e78MnTRjCFGKp/SEE?node-id=3097-14775\n"
         "Let me know if I've missed anything or if you need additional design references.")

ONE_LINE = ("Hi team, here's the link to the complete user flow for a group call: "
            "https://www.figma.com/design/DVFotKid0e78MnTRjCFGKp/SEE?node-id=3097-14775 "
            "Let me know if I've missed anything or if you need design references.")
BLOG = ("Good morning everyone, hope you all had a good weekend. I stumbled onto this "
        "blog post about weird pagination patterns and I can't stop thinking about it, "
        "linked below, no action needed I just think you'll enjoy it: "
        "https://example.com/pagination Also someone left a birthday cake in the kitchen.")
LOLCAT = ("lol check this out https://youtube.com/watch?v=abc and this one "
          "https://youtube.com/watch?v=xyz haha so good I watched both twice last night")

# (name, callable -> bool, expected)
CARDS = [
    ("a build link message cards",            lambda: S.summarize(BUILD) is not None, True),
    ("a cannot-list message cards",           lambda: S.summarize(BUGS) is not None, True),
    ("a described design link cards",         lambda: S.summarize(FIGMA) is not None, True),
    # The two-sentence floor still applies to a link message. One run-on line is a single
    # statement, and a summary of one statement is longer than the statement.
    ("a one-sentence link message does NOT card",
     lambda: S.summarize(ONE_LINE) is not None, False),
    ("a described BLOG link does NOT card",   lambda: S.summarize(BLOG) is not None, False),
    ("chatter with two links does NOT card",  lambda: S.summarize(LOLCAT) is not None, False),
]

SCORES = [
    ("'cannot set' scores",            lambda: S.importance("Cannot set group pictures") > 0, True),
    ("'can't stop' does NOT score",    lambda: S.importance("I can't stop thinking about it") > 0, False),
    ("'can't wait' does NOT score",    lambda: S.importance("I can't wait for the demo") > 0, False),
    ("plural 'issues' scores",         lambda: S.importance("The UI issues have been addressed") > 0, True),
    ("plural 'bugs' scores",           lambda: S.importance("Two bugs are still open") > 0, True),
    ("'addressed' scores",             lambda: S.importance("Both have been addressed") > 0, True),
    ("'take a decision' scores",       lambda: S.importance("I want to take a decision on Ram") > 0, True),
    ("'does that work' scores",        lambda: S.importance("Does that work for everyone") > 0, True),
    ("'I wanted a call to decide' survives",
     lambda: S.importance("I wanted a call to take a decision on Ram which is 4GB") > 0, True),
    ("'I wanted to vent' still cancelled",
     lambda: S.importance("I wanted to vent about the coffee machine") > 0, False),
]

# (name, source, bullets in, must appear in the result)
LINKS = [
    ("a build url reaches the card", BUILD,
     ["Use the following build for testing"], "Build: https://install.shareipa.com/ccUH8I"),
    ("a figma url is labelled Design", FIGMA,
     ["Link to complete user flow"], "Design: https://www.figma.com/design"),
    ("a url already in a bullet is not repeated",
     "see https://install.shareipa.com/ccUH8I for the build",
     ["Build at https://install.shareipa.com/ccUH8I"], None),
    ("no url, nothing added", "Login works and the chat list scrolls",
     ["Login works on both platforms"], None),
]

# The vocative cases that used to live here were removed with the mechanism on
# 2026-10-11 - see the note above _restore_links in summary_engines.py. They asserted that
# a capitalised word against a comma was a person, which measurement showed deletes
# product names and owners ("on Chrome,", "owner: Mike,"). Nothing asserts that now
# BECAUSE nothing should: a name occasionally reaching a title is the accepted failure
# until the backend sends the room's participant names.


def main():
    failed = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed.append(name)
            print(f"        got {got!r}, want {want!r}")

    print("\n  does the message earn a card")
    for name, fn, want in CARDS:
        check(name, fn(), want)

    print("\n  importance scoring")
    for name, fn, want in SCORES:
        check(name, fn(), want)

    print("\n  urls reach the card")
    for name, src, points, must in LINKS:
        out = _restore_links(src, points)
        if must is None:
            check(name, out, points)                 # unchanged
        else:
            check(name, any(p.startswith(must[:24]) for p in out), True)

    total = len(CARDS) + len(SCORES) + len(LINKS)
    print(f"\n  TOTAL {total - len(failed)}/{total} correct")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
