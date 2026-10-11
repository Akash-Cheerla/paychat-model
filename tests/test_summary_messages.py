"""Does the Summary card keep the important parts and drop the rest?

Akash, 2026-10-08: "summary is to summarize important parts of messages not everything".
Fourteen hand-written cases (tests/test_summarizer.py) say the obvious ones work; they
cannot say whether the thing that matters survives a long rambling message, which is the
only kind anyone needs a summary of. This runs the generated set, where every message
carries the parts that must survive and the parts that must not.

Three numbers, and they trade off against each other:

  KEPT      of everything marked important, how much reached the card. Missing a deadline
            is the failure that makes the tab useless.
  LEAKED    of everything marked chatter, how much reached the card anyway. This is the
            "not everything" half - a card that reprints the message is not a summary.
  NO CARD   messages that are pure chatter and correctly produced nothing.

What the generated labels are good for and what they are not: "this sentence contains the
deadline" is close to objective and the generator is reliable at it. "This sentence is
chatter" is a judgement, and a borderline call ("Login is fine on both platforms" - status
or noise?) is the generator's opinion, not ground truth. So LEAKED is read as a trend and
a list to eyeball, never as a pass/fail bar. KEPT is the number to hold the line on.

Matching is on content words, not substrings: a bullet is deliberately rewritten from the
sentence it came from ("Please upload your evidence" -> "Upload your evidence"), so an
exact-substring test would report losses that are not losses.

    python tests/test_summary_messages.py
    python tests/test_summary_messages.py --show-missed
"""
import argparse
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from summarizer import summarize   # noqa: E402
import summary_engines            # noqa: E402

# Words that carry no identity - present in almost every sentence, so matching on them
# would make everything look covered.
_STOP = frozenset("""
a an the this that these those it its we you they them our your their my his her i me
in on at to for of with from by as into onto over under after before once when while
until during about across through between among per via and or but not no nor if then
is are was were be been being has have had do does did will would shall should can
could may might must need needs so because there here just also still only very really
please kindly make sure
""".split())


def content_words(text):
    words = re.findall(r"[a-z0-9][a-z0-9'\-\.]*", (text or "").lower())
    return [w for w in words if w not in _STOP and len(w) > 1]


def coverage(phrase, points):
    """The best fraction of `phrase`'s content words found in any single point."""
    want = content_words(phrase)
    if not want:
        return 1.0
    best = 0.0
    for p in points:
        have = set(content_words(p))
        best = max(best, sum(1 for w in want if w in have) / len(want))
    return best


KEEP_BAR = 0.7      # most of the phrase present in one bullet = it survived
DROP_BAR = 0.8      # nearly all of it present = the chatter was reprinted, not incidental


def main():
    ap = argparse.ArgumentParser()
    # Accepted and ignored: this runs entirely offline, but tests/run_all.py hands every
    # battery a --url and argparse would otherwise reject the whole run.
    ap.add_argument("--url", help=argparse.SUPPRESS)
    ap.add_argument("--name", help=argparse.SUPPRESS)
    ap.add_argument("--cases", default="data/eval/summary_messages.json")
    ap.add_argument("--show-missed", action="store_true")
    ap.add_argument("--show-leaked", action="store_true")
    # The rules are only half of what ships: production_summary() uses them as the gate
    # and the student writes the card. Without this the battery scores an engine no
    # caller gets, which is how the KEPT number came to be quoted for the rules alone.
    ap.add_argument("--production", action="store_true",
                    help="score the production path (rules gate + student) instead")
    a = ap.parse_args()

    cases = json.loads((ROOT / a.cases).read_text(encoding="utf-8"))

    if a.production:
        student = summary_engines.default_student()
        if student is None:
            print("  no student weights on this box - cannot score the production path")
            return 1
        print(f"  scoring the PRODUCTION path (rules gate + student) - "
              f"~3s a card, {len(cases)} messages")

        def card_for(text):
            out = summary_engines.production_summary(text, student)
            if out is None:
                return None
            return type("D", (), {"points": out["points"], "title": out["title"]})()
    else:
        print("  scoring the RULES only - pass --production for what ships")
        card_for = summarize
    kept = missed = leaked = clean = 0
    chatter_right = chatter_total = 0
    no_card = 0
    missed_rows, leaked_rows, wrong_chatter = [], [], []

    for c in cases:
        d = card_for(c["text"])
        points = d.points if d else []
        if d is None:
            no_card += 1

        if not c["must_keep"]:
            # Pure chatter: the right answer is no card at all.
            chatter_total += 1
            if d is None:
                chatter_right += 1
            else:
                wrong_chatter.append((c["name"], d.title, points))
            continue

        for phrase in c["must_keep"]:
            if coverage(phrase, points) >= KEEP_BAR:
                kept += 1
            else:
                missed += 1
                missed_rows.append((c["name"], phrase, points))
        for phrase in c["must_drop"]:
            if coverage(phrase, points) >= DROP_BAR:
                leaked += 1
                leaked_rows.append((c["name"], phrase, points))
            else:
                clean += 1

    total_keep = kept + missed
    total_drop = leaked + clean
    pct = lambda n, d: (100.0 * n / d) if d else 0.0      # noqa: E731

    print(f"\n  TOTAL {kept}/{total_keep} important parts kept")
    print(f"  {len(cases)} messages, {no_card} produced no card")
    print(f"    KEPT    {kept}/{total_keep}  ({pct(kept, total_keep):.1f}%)  "
          f"- the number to hold the line on")
    print(f"    LEAKED  {leaked}/{total_drop}  ({pct(leaked, total_drop):.1f}%)  "
          f"- chatter that reached the card")
    if chatter_total:
        print(f"    CHATTER {chatter_right}/{chatter_total}  "
              f"({pct(chatter_right, chatter_total):.1f}%)  - all-noise messages correctly dropped")

    if wrong_chatter:
        print(f"\n  {len(wrong_chatter)} all-chatter message(s) that still made a card:")
        for name, title, pts in wrong_chatter[:8]:
            print(f"    {name}\n        {title}: {pts}")
    if missed_rows and (a.show_missed or len(missed_rows) <= 12):
        print(f"\n  {len(missed_rows)} important part(s) missed:")
        for name, phrase, pts in missed_rows[:25]:
            print(f"    {name}\n        wanted: {phrase[:88]}")
            print(f"        card:   {pts if pts else 'NO CARD'}")
    elif missed_rows:
        print(f"\n  {len(missed_rows)} missed - rerun with --show-missed")
    if leaked_rows and a.show_leaked:
        print(f"\n  {len(leaked_rows)} chatter phrase(s) that leaked:")
        for name, phrase, _ in leaked_rows[:25]:
            print(f"    {name}: {phrase[:88]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
