"""
Lightweight keyword trigger filter — pre-screens messages before they hit
the model. If a message has no action-related words, skip it entirely.

This is intentionally high-recall, low-precision. It catches anything that
MIGHT be actionable. The model does the real classification. The point is
to filter out the 80% of casual chat ("lol", "haha", "bro that was insane")
so the model/enclave never even sees them.

Usage:
    from trigger_filter import should_process, matched_intents

    if should_process("venmo me $25"):
        result = run_model(["venmo me $25"])   # worth classifying
    else:
        pass  # casual chat, skip

    matched_intents("venmo me $25")  # -> ["money"]  (which groups matched)
"""
from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# Per-intent keyword sets. A message triggers if ANY keyword matches.
# These are deliberately broad — false positives are fine, the model filters.
# ─────────────────────────────────────────────────────────────────────────────

_TRIGGERS: dict[str, list[str]] = {
    "money": [
        r"\$\d", r"₹\d", r"€\d",
        r"\bpay\w*\b", r"\bpaid\b", r"\bowe[sd]?\b", r"\bdebt\b", r"\bloan\b",
        r"\bvenmo\b", r"\bcashapp\b", r"\bcash app\b", r"\bzelle\b",
        r"\bpaypal\b", r"\bpaytm\b", r"\bg[\- ]?pay\b", r"\bphonepe\b",
        r"\bupi\b", r"\bsplit\b", r"\bsettle\b",
        r"\bbucks?\b", r"\bdollars?\b", r"\brupees?\b",
        r"\bsend me\b", r"\bpay me\b", r"\bpay back\b",
        r"\btransfer\b", r"\breimburse\b", r"\bcover me\b", r"\bspot me\b",
        r"\blend me\b",
    ],
    "alarm": [
        r"\bremind\b", r"\breminder\b", r"\balarm\b", r"\btimer\b",
        r"\bwake me\b", r"\bping me\b", r"\bnotify me\b",
        r"\bdon'?t forget\b", r"\bdon'?t let me forget\b",
    ],
    "contact": [
        r"\bcall\b", r"\btext\b", r"\bdial\b", r"\bphone\b",
        r"\bwhatsapp\b", r"\bfacetime\b", r"\bmessage\b",
        r"\bsave.{0,10}number\b", r"\bsave.{0,10}contact\b",
        r"\+\d{1,3}\s?\d", r"\(\d{3}\)",
    ],
    "calendar": [
        r"\bmeeting\b", r"\bschedule\b", r"\bappointment\b",
        r"\bstandup\b", r"\bsync\b", r"\binterview\b",
        r"\bdinner\b", r"\blunch\b", r"\bbrunch\b",
        r"\b\d{1,2}\s?[ap]m\b", r"\bo'?clock\b",
        r"\btomorrow\b", r"\bmonday\b", r"\btuesday\b", r"\bwednesday\b",
        r"\bthursday\b", r"\bfriday\b", r"\bsaturday\b", r"\bsunday\b",
    ],
    "maps": [
        r"\bdirections?\b", r"\bnavigate\b", r"\bheading to\b",
        r"\bomw\b", r"\bon my way\b", r"\bwhere is\b", r"\bwhere'?s\b",
        r"\bmeet me at\b", r"\bmeet at\b", r"\baddress\b",
        r"\bgoogle maps\b", r"\bapple maps\b", r"\bwaze\b",
    ],
    "ride": [
        r"\buber\b", r"\blyft\b", r"\bola\b", r"\bcab\b", r"\btaxi\b",
        r"\bride\b", r"\bbook.{0,10}ride\b", r"\bget.{0,10}cab\b",
    ],
    "food_order": [
        r"\border\b", r"\bdeliver\b", r"\bdelivery\b",
        r"\bswiggy\b", r"\bzomato\b", r"\bdoordash\b", r"\bubereats\b",
        r"\bgrubhub\b", r"\bpizza\b", r"\bburger\b", r"\bfood\b",
        r"\bbiryani\b", r"\bnoodles\b", r"\bsushi\b",
    ],
    "travel": [
        r"\bflight\b", r"\bflights\b", r"\bhotel\b", r"\bairbnb\b",
        r"\bbooking\b", r"\btrip\b", r"\btravel\b",
        r"\btrain ticket\b", r"\bbus ticket\b",
        r"\bmakemytrip\b", r"\bexpedia\b", r"\bkayak\b",
    ],
    "shopping": [
        r"\bbuy\b", r"\bpurchase\b", r"\bamazon\b", r"\bflipkart\b",
        r"\bcart\b", r"\border.{0,15}online\b",
        r"\badd to cart\b",
    ],
    "music": [
        r"\bplay\b", r"\bsong\b", r"\bmusic\b", r"\bplaylist\b",
        r"\bspotify\b", r"\bapple music\b", r"\byoutube music\b",
        r"\bgaana\b", r"\bjiosaavn\b",
    ],
    "video": [
        r"\bwatch\b", r"\bnetflix\b", r"\byoutube\b", r"\bstream\b",
        r"\bhulu\b", r"\bprime video\b", r"\bdisney\b", r"\bhotstar\b",
        r"\bput on\b", r"\bmovie\b", r"\bshow\b", r"\bepisode\b",
    ],
    "tickets": [
        r"\btickets?\b", r"\bconcert\b", r"\bbookmyshow\b", r"\bfandango\b",
        r"\bshow\b", r"\bevent\b",
    ],
    "reservation": [
        r"\breserv\w*\b", r"\btable for\b", r"\bbook.{0,10}table\b",
        r"\bopentable\b", r"\bdineout\b", r"\bparty of\b",
    ],
    "task": [
        r"\btodo\b", r"\bto-do\b", r"\btask\b", r"\badd to.{0,10}list\b",
        r"\btodoist\b",
    ],
    "note": [
        r"\bnote:", r"\bnote that\b", r"\bjot\b", r"\bwrite down\b",
        r"\bnote to self\b",
    ],
    "bills": [
        r"\bbill\b", r"\belectricity\b", r"\brent\b", r"\bwifi\b",
        r"\binternet\b", r"\bdue\b", r"\butility\b", r"\brecharge\b",
        r"\binsurance\b",
    ],
    "health": [
        r"\bdoctor\b", r"\bdentist\b", r"\bhospital\b", r"\bclinic\b",
        r"\bpharmacy\b", r"\bpracto\b", r"\bzocdoc\b",
        r"\bappointment\b", r"\bcheckup\b", r"\bcheck-up\b",
    ],
    "weather": [
        r"\bweather\b", r"\brain\b", r"\btemperature\b", r"\bforecast\b",
        r"\bsunny\b", r"\bcloudy\b", r"\bhumidity\b",
        r"\bhow.{0,5}(hot|cold)\b",
    ],
}

# Compile all patterns once at import time
_COMPILED: dict[str, re.Pattern] = {
    intent: re.compile("|".join(patterns), re.IGNORECASE)
    for intent, patterns in _TRIGGERS.items()
}


def matched_intents(text: str) -> list[str]:
    """Return which intent groups have keyword matches in the text."""
    return [intent for intent, pat in _COMPILED.items() if pat.search(text)]


def should_process(text: str) -> bool:
    """True if the message has any action-related keywords worth classifying."""
    return any(pat.search(text) for pat in _COMPILED.values())


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # should trigger
        ("venmo me $25", True, ["money"]),
        ("remind me at 6pm", True, ["alarm"]),
        ("call dad", True, ["contact"]),
        ("meeting at 3pm tomorrow", True, ["calendar"]),
        ("heading to SFO", True, ["maps"]),
        ("book an uber", True, ["ride"]),
        ("order pizza from dominos", True, ["food_order"]),
        ("flights to goa", True, ["travel"]),
        ("buy 2 hdmi cables on amazon", True, ["shopping"]),
        ("play anti-hero on spotify", True, ["music"]),
        ("watch the office on netflix", True, ["video"]),
        ("2 tickets for oppenheimer", True, ["tickets"]),
        ("table for 4 at olive garden", True, ["reservation"]),
        ("add groceries to my todo", True, ["task"]),
        ("note: meeting moved to 3pm", True, ["note"]),
        ("pay electricity bill", True, ["money", "bills"]),
        ("book a dentist appointment", True, ["health"]),
        ("will it rain tomorrow", True, ["weather"]),
        ("can we settle the loan which I'm owed by you", True, ["money"]),

        # should NOT trigger
        ("lol", False, []),
        ("haha nice", False, []),
        ("bro that was insane", False, []),
        ("what's up", False, []),
        ("good morning", False, []),
        ("yeah sure", False, []),
        ("ok cool", False, []),
        ("i'm so tired", False, []),
        ("that's crazy", False, []),
        ("ikr", False, []),
        ("omg really", False, []),
        ("bruh moment", False, []),
    ]

    passed = 0
    for text, expected_trigger, expected_intents in tests:
        result = should_process(text)
        intents = matched_intents(text)
        ok = result == expected_trigger
        if expected_intents:
            ok = ok and all(i in intents for i in expected_intents)
        status = "ok" if ok else "FAIL"
        if not ok:
            print(f"  {status}: '{text}' -> trigger={result} (expected {expected_trigger}), intents={intents}")
        passed += ok

    print(f"\ntrigger filter: {passed}/{len(tests)} passed")
