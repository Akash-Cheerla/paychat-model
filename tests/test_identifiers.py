"""Do ticket numbers survive into the card?

The student does not invent identifiers - 0 fabricated across 69 real cards - but it
omitted them. Measured 2026-10-11 over the parts the labels mark important:

    ticket id     8/20    40.0%     <- this
    version      19/24    79.2%
    weekday      65/77    84.4%
    clock time   29/35    82.9%
    ratio/size   14/16    87.5%
    amount        3/3    100.0%

For this team the ticket number is what someone opens the tab to find, so
_restore_ticket_ids() puts back any id that is in the message but missing from the card.
It does that two ways, because the drops had two causes: an id whose sentence clearly
produced a bullet is appended to that bullet, and an id sitting in a sentence that
carries no content of its own ("That's PAY-1198") goes on one trailing line.

The risk it carries is the reason for this file. A misattributed ticket number is worse
than a missing one, so the cases below are mostly about what it must NOT do: not attach
to a bullet that did not come from that sentence, not read a version or a date as a
ticket, not invent, not reorder, not drop. A future retrain or prompt change must not
quietly break any of it.

    python tests/test_identifiers.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from summary_engines import _restore_dates, _restore_ticket_ids   # noqa: E402

# Dates are restored by a SEPARATE mechanism, deliberately. An id is opaque and can only
# belong to one thing; a date is semantic, and reusing the ticket logic put two
# contradictory deadlines on one bullet. Of the 27 dropped date tokens, 14 looked safely
# placeable by overlap alone and would have produced "Fix needed by Wednesday EOD
# (Thursday)", "...report blockers by Thursday (EOD)" and the same figure twice as "(4.5)"
# and "(4.5:1)". The two guards below are what the cases here hold.
#
# (name, source message, bullets in, expected bullets out)
DATES = [
    ("places a deadline on the bullet from its sentence",
     "Ops needs a dry run on the staging clone before Friday at noon.",
     ["Ops needs dry run on staging clone"],
     ["Ops needs dry run on staging clone (Friday)"]),

    ("REFUSES a bullet that already carries a deadline",
     "Fix needed by Wednesday EOD. Please also read the notes before Thursday.",
     ["Fix needed by Wednesday EOD"],
     ["Fix needed by Wednesday EOD"]),

    ("collapses the same figure matched by two patterns",
     "The color palette fails the contrast check, it needs at least 4.5:1 against "
     "the background.",
     ["Color palette fails contrast check"],
     ["Color palette fails contrast check (4.5:1)"]),

    ("collapses 09:15 UTC rather than adding 09:15 as well",
     "Database CPU spiked to 100% at 09:15 UTC during the incident.",
     ["Database CPU spiked to 100 percent"],
     ["Database CPU spiked to 100 percent (09:15 UTC)"]),

    ("does not attach to a bullet from a different sentence",
     "We should push the Friday release to Tuesday. Priya will take the gift card crash.",
     ["Priya to take gift card crash"],
     ["Priya to take gift card crash"]),

    ("a version reaches the bullet it belongs to",
     "Login works on Android and push works. Both are punted to 2.4.1.",
     ["Both punted to the next release"],
     None),            # asserted on count only - placement depends on overlap

    ("no dates in the message changes nothing",
     "Login works on both platforms and the chat list scrolls",
     ["Login works on both platforms"],
     ["Login works on both platforms"]),

    ("a date already in the card is not repeated",
     "Ship it on Friday please.",
     ["Ship on Friday"],
     ["Ship on Friday"]),
]

# (name, source message, bullets in, expected bullets out)
CASES = [
    ("attaches to the bullet that came from the sentence",
     "There is a hard crash on iOS 17.2 when you apply a gift card before adding a "
     "shipping address, logged as JIRA-4471. The order summary card also overflows on "
     "iPhone SE.",
     ["Crash on iOS 17.2 when applying gift card",
      "Order summary card overflows on iPhone SE"],
     ["Crash on iOS 17.2 when applying gift card (JIRA-4471)",
      "Order summary card overflows on iPhone SE"]),

    ("never attaches to a bullet from a different sentence",
     "Priya flagged a checkout failure in the staging build, see CHECKOUT-4417. The "
     "coffee machine on our floor is broken again.",
     ["Coffee machine is broken again"],
     ["Coffee machine is broken again", "Ticket: CHECKOUT-4417"]),

    ("a back-reference goes to the trailing line, not a guessed bullet",
     "The currency symbol is wrong for JPY on the payment sheet. That's PAY-1198.",
     ["Currency symbol wrong for JPY on payment sheet"],
     ["Currency symbol wrong for JPY on payment sheet", "Ticket: PAY-1198"]),

    ("ids logged together land on one line, not spread over bullets",
     "Login fails on Android. The chat list jumps. The amount shows blank. I've logged "
     "all of these in JIRA under QA-101, QA-102, and QA-103.",
     ["Login fails on Android", "Chat list jumps", "Amount shows blank"],
     ["Login fails on Android", "Chat list jumps", "Amount shows blank",
      "Tickets: QA-101, QA-102, QA-103"]),

    ("leaves an id that is already in the card alone",
     "QA-101 is still failing on the latest build.",
     ["QA-101 is still failing today"],
     ["QA-101 is still failing today"]),

    ("does not mistake a version for a ticket",
     "We are shipping 3.2.0-rc3 on Friday and 4.5:1 is the contrast floor.",
     ["Build passed the full regression suite"],
     ["Build passed the full regression suite"]),

    ("does not mistake a date for a ticket",
     "The cutover is locked for 2026-10-18 at 02:00 UTC.",
     ["Cutover locked for the 18th at 02:00 UTC"],
     ["Cutover locked for the 18th at 02:00 UTC"]),

    ("two ids from two sentences reach their own bullets",
     "Staging DB is dropping connections under load, that is INFRA-991. Separately we "
     "need to add schema validation at the ingest boundary, PLAT-4471.",
     ["Staging DB dropping connections under load",
      "Add schema validation at ingest boundary"],
     ["Staging DB dropping connections under load (INFRA-991)",
      "Add schema validation at ingest boundary (PLAT-4471)"]),

    ("two ids on one bullet read as one reference",
     "The staging DB keeps dropping connections under load: DB-1, DB-2, DB-3, DB-4 are "
     "all the same root cause.",
     ["Staging DB keeps dropping connections under load"],
     ["Staging DB keeps dropping connections under load (DB-1, DB-2)",
      "Tickets: DB-3, DB-4"]),

    ("a long list of ids is capped with a count",
     "Everything is logged. See AA-1, BB-2, CC-3, DD-4, EE-5, FF-6, GG-7, HH-8.",
     ["Several unrelated issues were filed"],
     ["Several unrelated issues were filed",
      "Tickets: AA-1, BB-2, CC-3, DD-4, EE-5, FF-6 and 2 more"]),

    ("empty card stays empty - no card is better than a card of ids",
     "See QA-500 for details.",
     [],
     []),

    ("no ids in the message changes nothing",
     "Login works on both platforms now and the chat list scrolls properly.",
     ["Login works on both platforms", "Chat list scrolls properly"],
     ["Login works on both platforms", "Chat list scrolls properly"]),
]


def main():
    failed = []
    for name, src, points, want in CASES:
        before = list(points)
        got = _restore_ticket_ids(src, points)
        ok = got == want

        # invariants that hold for every case, whatever the expectation says
        if ok:
            if points != before:
                ok, why = False, "mutated the caller's list"
            elif len(got) not in (len(before), len(before) + 1):
                ok, why = False, "changed the bullet count by more than the one line"
            elif [p for p in got if not p.startswith(("Ticket:", "Tickets:"))] and not all(
                    b.split(" (")[0] in " ".join(got) for b in before):
                ok, why = False, "lost or rewrote an existing bullet"

        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed.append(name)
            print(f"        got:  {got}")
            print(f"        want: {want}")

    print("\n  deadlines, versions and measurements")
    for name, src, points, want in DATES:
        before = list(points)
        got = _restore_dates(src, points)
        if want is None:
            # placement depends on overlap; what must hold is that nothing is invented,
            # nothing is lost, and no bullet ends up with two times on it
            ok = (len(got) == len(before)
                  and all(got[i].startswith(before[i]) for i in range(len(before))))
        else:
            ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed.append(name)
            print(f"        got:  {got}")
            print(f"        want: {want}")

    total = len(CASES) + len(DATES)
    print(f"\n  TOTAL {total - len(failed)}/{total} correct")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
