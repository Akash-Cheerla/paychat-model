"""What must never reach a summary card, in one place so the two paths cannot drift.

Both summarisers persist what they produce and show it again later, possibly on a shared
screen or in a group. So a card must never carry a credential or somebody's identifier,
and this is the single definition of that - imported by summary_engines (per-message) and
room_summarizer (per-window) rather than written twice.

Measured 2026-10-17 against a corpus planted with 50 leaks in 1v1 and group rooms. Before
this existed the per-message path produced, verbatim:

    "Rupesh sent OTP 448201 over chat"
    "Password is hunter2monkey"
    "Company card 4111 1111 1111 1111"
    "Aadhaar 4821 9930 1174 requested"
    "Api key sk-live-9f2b8a71c044e5 worked yesterday"
    "Blood test abnormal" / "Doctor booked scan for Thursday"

The conversation path already refused all of them, which is why this is the per-message
path's pattern moved out rather than a new idea.

A point is dropped, not the whole card: "Build 61.3 up on TestFlight" and "Verify payment
sheet before Friday" came from the same message as the OTP and are worth keeping. A card
left with nothing is no card at all.
"""
import re

# Credentials and identifiers. The leak is usually the VALUE next to the word, but the
# word alone is enough to refuse on - a card that says "the password is" has already told
# the reader where to look.
SECRET = re.compile(
    r"\b(?:otp|one[\s-]?time[\s-]?(?:code|password|pin)|password|passcode|passphrase|"
    r"pwd|cvv|cvc|pin\s*(?:code|number|is)?|2fa|mfa|two[\s-]?factor|"
    r"aadhaar|aadhar|pan\s*card|ssn|passport\s*(?:no|number)|"
    r"ifsc|upi\s*(?:id|pin)|api[_\s-]?key|secret\s*key|access\s*token|auth\s*token|"
    r"bearer\s+[\w.-]{8,}|sk-[\w-]{8,}|"
    r"account\s*(?:no|num|number)|card\s*(?:no|num|number)|verification\s*code)\b"
    r"|\b\d{10,}\b"                        # phone numbers and long identifiers
    r"|\b\d{4}[\s-]?\d{4}[\s-]?\d{4}(?:[\s-]?\d{4})?\b",   # card-shaped digit groups
    re.IGNORECASE)

# Health is the other thing that must not be restated in a card someone else may read.
# Deliberately narrow: a clinical fact about a person, not the word "sick" in passing.
HEALTH = re.compile(
    r"\b(?:blood\s*test|biopsy|scan|x-?ray|mri|ct\s*scan|ultrasound|"
    r"diagnos\w*|prescription|medication|meds|dosage|chemo\w*|surgery|operation|"
    r"hospital|doctor|physician|clinic|therapist|therapy|psychiatr\w*|"
    r"pregnan\w*|miscarriage|depress\w*|anxiety\s*(?:attack|disorder|meds)?)\b",
    re.IGNORECASE)


def is_sensitive(text: str) -> bool:
    """Should this never appear on a card?"""
    t = text or ""
    return bool(SECRET.search(t) or HEALTH.search(t))


def clean_card(card):
    """Drop the points that must not be shown; None if nothing survives.

    Takes and returns the {"title", "points"} shape both engines use. A sensitive TITLE
    kills the card outright - there is nowhere to put it and no way to show the card
    without it.
    """
    if not card:
        return card
    if is_sensitive(card.get("title", "")):
        return None
    points = [p for p in card.get("points") or [] if not is_sensitive(p)]
    if not points:
        return None
    return {**card, "points": points}
