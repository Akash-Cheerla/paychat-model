"""Nothing that must stay in the chat reaches a card, on either summary path.

Both summarisers persist what they produce and show it again later, possibly in a group or
on a shared screen. Measured 2026-10-17 against a corpus planted with 50 leaks across 26
1v1 and 8 group rooms, the per-message path produced these VERBATIM:

    "Rupesh sent OTP 448201 over chat"
    "Password is hunter2monkey"
    "Company card 4111 1111 1111 1111"
    "Aadhaar 4821 9930 1174 requested"
    "Api key sk-live-9f2b8a71c044e5 worked yesterday"
    "Blood test abnormal" / "Doctor booked scan for Thursday"

and the intent path put a card number into the phone slot of a money prompt:

    "I used the company card 4111 1111 1111 1111 for the dinner, send me your share"
      -> slots: {"phone": "4111111111111111"}

sensitive.py is the single definition both summarisers use, so they cannot drift. The
intent path has its own fix in app.py, because a phone slot is not a card.

The REFUSALS are only half of this file. A guard that also eats "can you send me 500 for
the cab" has broken the product, so every refusal here is paired with the money or ride
message that has to survive it.

    python tests/test_no_leaks.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sensitive   # noqa: E402

# (name, text, must be refused)
GUARD = [
    ("an OTP",                  "the otp is 448201, use it before it expires", True),
    ("an OTP request",          "Can you share the OTP for the portal please", True),
    ("a password",              "my password for the staging box is hunter2monkey", True),
    ("a wifi password",         "can you change the wifi password, everyone has it", True),
    ("a card PIN",              "the card pin is 4417, dont tell anyone", True),
    ("a CVV",                   "cvv is 319 and expiry is 08/29", True),
    ("an API key",              "here is the api key sk-live-9f2b8a71c044e5", True),
    ("a UPI pin",               "my upi pin stopped working after the update", True),
    ("a UPI id",                "send it to my upi id rupesh@okhdfc", True),
    ("a UPI handle with digits","pay 9845731890@ybl please", True),
    ("an aadhaar",              "they need my aadhaar 4821 9930 1174 for the kyc", True),
    ("a phone number",          "add this number to the group 9845731890", True),
    ("an account number",       "account number 00123456789 and ifsc HDFC0001234", True),
    ("a card number",           "the card is 4111 1111 1111 1111 if you want to book", True),
    ("a clinical result",       "my blood test came back and the doctor wants a scan", True),
    ("medication",              "started the new medication yesterday, makes me drowsy", True),
    ("a hospital day",          "her surgery is on thursday so i will be at the hospital", True),
    ("therapy",                 "been seeing a therapist for the anxiety", True),

    # Paired survivors. If any of these flips to refused, the product is broken.
    ("a money request survives",      "can you send me 500 for the cab", False),
    ("a split survives",              "dinner was 2400 total, can everyone send their shares", False),
    ("a deadline survives",           "I need to leave by 4 so book it before that", False),
    ("a ride request survives",       "can you book a cab from my home to the airport", False),
    ("a rent request survives",       "I need 650 for my rent this month, can you help", False),
    ("a repayment promise survives",  "will send you back once i get paid on friday", False),
    ("a ticket id survives",          "QA-116 still fails on the latest build", False),
    ("a build number survives",       "Build 61.3 is on TestFlight, please pull it", False),
    ("a bug report survives",         "the split amount shows blank when participants is not sent", False),
    ("a payment-method status survives",
     "Failed because upi is not set on your end", False),
]

# clean_card drops the offending point, not the whole card - the clean points from the
# same message are worth keeping.
CARDS = [
    ("a sensitive point is dropped and the rest kept",
     {"title": "Build 61.3 TestFlight",
      "points": ["Build 61.3 up on TestFlight", "Rupesh sent OTP 448201 over chat",
                 "Verify payment sheet before Friday"]},
     {"title": "Build 61.3 TestFlight",
      "points": ["Build 61.3 up on TestFlight", "Verify payment sheet before Friday"]}),
    ("a card of nothing but a secret is dropped entirely",
     {"title": "Staging Handover", "points": ["Password is hunter2monkey"]}, None),
    ("a sensitive TITLE kills the card",
     {"title": "OTP 448201", "points": ["Build is up", "Please test it"]}, None),
    ("a clean card is untouched",
     {"title": "Build 61 Regression",
      "points": ["Login works on both platforms", "Chat list scrolls properly"]},
     {"title": "Build 61 Regression",
      "points": ["Login works on both platforms", "Chat list scrolls properly"]}),
    ("None in, None out", None, None),
]


def main():
    failed = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed.append(name)
            print(f"        got {got!r}")
            print(f"        want {want!r}")

    print("\n  what must never reach a card, and what must survive the guard")
    for name, text, want in GUARD:
        check(name, sensitive.is_sensitive(text), want)

    print("\n  the card is pruned, not thrown away")
    for name, card, want in CARDS:
        check(name, sensitive.clean_card(card), want)

    print("\n  the intent path does not read a card number as a phone number")
    import app   # noqa: E402  - imported late; loading it is slow
    PHONES = [
        ("a 16-digit card is refused", "I used the company card 4111 1111 1111 1111", None),
        ("a bare valid card is refused", "put it on 4111111111111111 please", None),
        ("another valid card is refused", "5500 0000 0000 0004 is on file", None),
        ("an aadhaar is refused", "they need my aadhaar 4821 9930 1174", None),
        ("an account number is refused", "account number 00123456789 for the transfer", None),
        ("a real mobile survives", "call me on 9845731890", "9845731890"),
        ("a country code survives", "reach me at +91 98457 31890", "+919845731890"),
        ("a US number survives", "my number is 415 555 0172", "4155550172"),
        ("an order id is still refused", "order 19101 has not shipped", None),
        ("a number after an order id is still found",
         "order 19101, call me on 9876543210", "9876543210"),
    ]
    for name, text, want in PHONES:
        check(name, app._extract_phone(text), want)

    total = len(GUARD) + len(CARDS) + len(PHONES)
    print(f"\n  TOTAL {total - len(failed)}/{total} correct")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
