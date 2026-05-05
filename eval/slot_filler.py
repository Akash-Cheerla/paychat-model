"""
v4 slot filler — Layer A (regex + dateparser).

Given (text, fired_intents) returns per-intent slots:

    extract_slots("call dad at 6pm", ["contact", "alarm"])
    -> {
        "contact": {"name": "dad", "channel": "call", "_required_filled": True},
        "alarm":   {"datetime": "2026-05-01T18:00:00", "note": "call dad",
                    "_required_filled": True},
    }

Design notes
------------
* Pure Python + dateparser. No network, no LLM, <5ms per message on CPU.
* Per the v4 spec, this is the first of three layers:
      A. Deterministic regex/dateparser  (this file)        ← cheap, ~70% coverage
      B. LLM extractor for the remaining ~30%               ← gpt-4o-mini, ~$0.0002/call
      C. Context resolver against backend tables            ← contacts, places, etc.
* Each intent's `SLOT_SCHEMA` lists `required` (must be filled or trigger clarification)
  and `optional` (nice to have). We never let `required` be `null` — if Layer A can't
  fill it, the result has `_required_filled = False` and the caller is expected to
  invoke Layer B or surface a clarification card to the user.

This module is import-clean — exposes:
    SLOT_SCHEMA               (dict)
    extract_slots(text, intents)  -> dict[intent, dict]
    needs_clarification(slots)    -> list[(intent, missing_slot)]
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import dateparser

# ─────────────────────────────────────────────────────────────────────────────
# Per-intent slot schema. `*` marks required slots — must always be filled or
# the action card cannot be rendered without a clarification step.
# ─────────────────────────────────────────────────────────────────────────────
SLOT_SCHEMA: dict[str, dict[str, list[str]]] = {
    "money":       {"required": ["amount", "recipient"],   "optional": ["note", "method", "direction"]},
    "alarm":       {"required": ["datetime"],              "optional": ["note", "recurrence"]},
    "contact":     {"required": ["name", "channel"],       "optional": ["note"]},
    "calendar":    {"required": ["title", "datetime"],     "optional": ["with_who", "location"]},
    "maps":        {"required": ["destination"],           "optional": ["origin", "mode"]},
    "ride":        {"required": ["destination"],           "optional": ["pickup", "when", "service"]},
    "food_order":  {"required": ["item"],                  "optional": ["provider", "when", "address"]},
    "travel":      {"required": ["destination"],           "optional": ["origin", "when", "mode"]},
    "shopping":    {"required": ["item"],                  "optional": ["qty", "provider", "when"]},
    "music":       {"required": [],                        "optional": ["song", "artist", "provider"]},
    "video":       {"required": [],                        "optional": ["title", "provider"]},
    "tickets":     {"required": ["event"],                 "optional": ["when", "qty"]},
    "reservation": {"required": ["place"],                 "optional": ["when", "qty"]},
    "task":        {"required": ["action"],                "optional": ["when"]},
    "note":        {"required": ["body"],                  "optional": []},
    "bills":       {"required": ["bill_kind"],             "optional": ["amount", "due"]},
    "health":      {"required": [],                        "optional": ["provider", "when", "kind"]},
    "weather":     {"required": [],                        "optional": ["place", "when"]},
}


# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns. Compiled once.
# ─────────────────────────────────────────────────────────────────────────────
_RE_AMOUNT = re.compile(
    r"(?:(?:\$|usd|inr|rs\.?|₹|€)\s?)?(\d+(?:[.,]\d{1,2})?)\s?"
    r"(?:dollars?|bucks?|usd|rs|rupees?|inr|euros?)?",
    re.IGNORECASE,
)
_RE_AMOUNT_STRICT = re.compile(
    r"(?:(?:\$|usd|inr|rs\.?|₹|€)\s?(\d+(?:[.,]\d{1,2})?)|"
    r"(\d+(?:[.,]\d{1,2})?)\s?(?:dollars?|bucks?|usd|rs|rupees?|inr|euros?))",
    re.IGNORECASE,
)
_RE_BARE_NUM_AMOUNT = re.compile(
    r"\b(\d{1,5}(?:[.,]\d{1,2})?)\b"
)
_RE_PHONE = re.compile(r"\+?\d[\d\s\-().]{7,}\d")
_RE_QTY = re.compile(r"\b(?:for|of|x)?\s*(\d{1,3})\s*(?:people|persons|pax|ppl|guests?|seats?)\b", re.IGNORECASE)

# channel detectors (contact intent)
_CHANNEL_PATTERNS = [
    (re.compile(r"\b(facetime|ft|face\s?time|video\s?call)\b", re.I),       "facetime"),
    (re.compile(r"\b(whatsapp|wa)\b", re.I),                                "whatsapp"),
    (re.compile(r"\b(imessage|sms|text|texting|message|messaging)\b", re.I),"text"),
    (re.compile(r"\b(call|calling|ring|phone|dial)\b", re.I),               "call"),
    (re.compile(r"\b(dm|email|emailing)\b", re.I),                          "dm"),
]

# names — dad/mom/etc. + capitalised tokens that follow a channel verb
_FAMILY_TERMS = {
    "dad", "mom", "mum", "mama", "papa", "father", "mother", "bro", "brother",
    "sis", "sister", "uncle", "aunt", "grandma", "grandpa", "grandmother",
    "grandfather", "wife", "husband", "boss", "kid", "son", "daughter",
    "boyfriend", "girlfriend", "bf", "gf", "friend",
}
_PRONOUNS_PEOPLE = {"him", "her", "them", "us"}  # need clarification ("who?")

# Channel verb followed by a name candidate
_RE_CHANNEL_TARGET = re.compile(
    r"\b(?:call|calling|ring|phone|dial|text|texting|message|messaging|"
    r"facetime|ft|whatsapp|sms|dm)\s+"
    r"(?P<target>[A-Za-z]+(?:\s+[A-Za-z]+)?)",
    re.IGNORECASE,
)

# "to <name>" / "<verb> <name> a text"
# Case-insensitive — chat is rarely properly capitalised. We exclude common
# pronouns ("me", "you") because the resolver handles those separately.
_RE_RECIPIENT_HINT = re.compile(
    r"\b(?:to|for|venmo|cashapp|zelle|paypal|paytm|gpay|phonepe)\s+"
    r"(?P<name>[A-Za-z][a-zA-Z]{1,30})\b",
    re.IGNORECASE,
)
_RECIPIENT_STOPLIST = {
    "me", "you", "us", "it", "now", "later", "back", "asap", "today", "tomorrow",
    "tonight", "yesterday", "the", "a", "an", "some", "more", "less",
    "rent", "lunch", "dinner", "coffee", "groceries", "pizza", "uber",
    "gas", "electricity", "internet", "water", "phone",
}

# location hints — "at <place>", "to <place>", "from <place>", "in <place>".
# We stop at filler words ("for", "with", time words, numbers) so we don't
# vacuum up the rest of the sentence.
_PLACE_STOP = (
    r"(?=\s+(?:at|on|in|for|with|to|from|by|tomorrow|today|tonight|tmrw|"
    r"yesterday|tonite|next|this|every|\d)|\s*$|[,.;!?])"
)
_RE_AT_PLACE = re.compile(
    rf"\b(?:at|@)\s+(?P<place>[A-Za-z][\w&'\-. ]{{1,40}}?){_PLACE_STOP}",
    re.IGNORECASE,
)
_RE_TO_PLACE = re.compile(
    rf"\bto\s+(?P<place>[A-Za-z][\w&'\-. ]{{1,40}}?){_PLACE_STOP}",
    re.IGNORECASE,
)
_RE_IN_PLACE = re.compile(
    rf"\bin\s+(?P<place>[A-Z][\w&'\-. ]{{1,40}}?){_PLACE_STOP}",
    re.IGNORECASE,
)
_AIRPORTS = {"jfk", "lga", "lax", "sfo", "ord", "atl", "bos", "sea", "iad", "dfw", "mia",
             "del", "bom", "blr", "maa", "hyd", "ccu", "lhr", "cdg", "fra", "sin", "hkg",
             "nrt", "hnd", "syd", "dxb", "doh"}

# bill kind detection
_BILL_KEYWORDS = {
    "rent": "rent", "mortgage": "mortgage", "electric": "electric",
    "electricity": "electric", "bijli": "electric", "water": "water",
    "internet": "internet", "wifi": "internet", "gas": "gas",
    "phone": "phone", "cellphone": "phone", "mobile": "phone",
    "credit card": "credit_card", "cc": "credit_card",
    "emi": "emi", "loan": "loan", "insurance": "insurance",
    "gym": "gym", "subscription": "subscription", "netflix": "subscription",
    "spotify": "subscription",
}

# food / shopping items rough capture (last-resort)
_RE_FOOD_HINT = re.compile(
    r"\b(?:order|doordash|ubereats|grubhub|swiggy|zomato|deliver|deliveroo)\s+"
    r"(?:some\s+|a\s+)?(?P<item>[\w& ]{2,30}?)(?=\s+(?:from|at|to|for|tonight|"
    r"tomorrow|today|on|by|via)|\s*$|[,.])",
    re.IGNORECASE,
)
_RE_SHOPPING_HINT = re.compile(
    r"\b(?:buy|order|get|pick up|grab)\s+(?:some\s+|a\s+|the\s+)?(?P<item>[\w& ]{2,30}?)"
    r"(?=\s+(?:from|on amazon|at|tonight|tomorrow|today|by|via)|\s*$|[,.])",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def _now() -> datetime:
    return datetime.now()


def _parse_when(text: str) -> str | None:
    """
    Extract a datetime from free text. Returns ISO-8601 or None.
    Handles 'tomorrow at 6pm', 'next friday', 'in 2 hours', '5am', 'mon 9:30',
    'kal subah 6 baje' (rough — falls back to dateparser).
    """
    # quick stop-words: pure pronouns / very short text → no datetime
    if not text or len(text) < 2:
        return None

    # Hindi-ish: kal/aaj/parso → tomorrow/today/day-after
    t = text
    t = re.sub(r"\bkal\b",  "tomorrow", t, flags=re.I)
    t = re.sub(r"\baaj\b",  "today",    t, flags=re.I)
    t = re.sub(r"\bparso\b", "day after tomorrow", t, flags=re.I)
    t = re.sub(r"\bsubah\b", "morning", t, flags=re.I)
    t = re.sub(r"\bshaam\b", "evening", t, flags=re.I)
    t = re.sub(r"\braat\b",  "night",   t, flags=re.I)
    t = re.sub(r"\bbaje\b",  "",        t, flags=re.I)

    parsed = dateparser.parse(
        t,
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": _now(),
        },
    )
    if parsed is None:
        # Try just lifting the time-shaped slice ("6pm tomorrow", "at 5am")
        m = re.search(
            r"(\b(?:on\s+)?(?:next\s+|this\s+)?(?:mon|tue|wed|thu|fri|sat|sun)\w*"
            r"(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)|"
            r"(\b(?:in\s+\d+\s+(?:min|minute|hour|day)s?))|"
            r"(\b(?:tomorrow|tonight|today|tmrw)(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)|"
            r"(\b\d{1,2}(?::\d{2})?\s*(?:am|pm))",
            t,
            re.IGNORECASE,
        )
        if m:
            parsed = dateparser.parse(
                m.group(0),
                settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": _now()},
            )
    if parsed is None:
        return None
    # If only a time was given, dateparser pins it to today; if that's already
    # past, push to tomorrow.
    if parsed < _now() - timedelta(minutes=1) and "ago" not in text.lower():
        parsed = parsed + timedelta(days=1)
    return parsed.replace(microsecond=0).isoformat()


def _extract_amount(text: str) -> str | None:
    m = _RE_AMOUNT_STRICT.search(text)
    if m:
        return (m.group(1) or m.group(2)).replace(",", "")
    # fall back to a bare number IF the message is clearly money-shaped
    if re.search(r"\b(venmo|cashapp|zelle|paypal|paytm|gpay|upi|phonepe|owe|"
                 r"transfer|send|pay|split|chip in)\b", text, re.I):
        m2 = _RE_BARE_NUM_AMOUNT.search(text)
        if m2:
            return m2.group(1)
    return None


def _extract_recipient(text: str) -> str | None:
    for m in _RE_RECIPIENT_HINT.finditer(text):
        cand = m.group("name").lower()
        if cand in _RECIPIENT_STOPLIST:
            continue
        return cand
    # family terms anywhere in the text (e.g. "send mom 50")
    for term in _FAMILY_TERMS:
        if re.search(rf"\b{term}\b", text, re.I):
            return term
    return None


def _extract_channel(text: str) -> str | None:
    for pat, label in _CHANNEL_PATTERNS:
        if pat.search(text):
            return label
    return None


def _extract_contact_name(text: str) -> str | None:
    """
    Returns a *resolvable* name string, e.g. 'dad', 'mom', 'sarah',
    or '__pronoun__' if the user said 'him/her/them' (caller must
    ask for clarification).
    """
    text_l = text.lower()

    # pronouns first (must trigger clarification, not extraction)
    for pron in _PRONOUNS_PEOPLE:
        if re.search(rf"\b{pron}\b", text_l):
            return "__pronoun__"

    # family terms
    for term in _FAMILY_TERMS:
        if re.search(rf"\b{term}\b", text_l):
            return term

    # channel verb followed by a likely name
    m = _RE_CHANNEL_TARGET.search(text)
    if m:
        cand = m.group("target").strip()
        # Filter out filler tokens captured after the verb
        skip = {"please", "back", "now", "later", "soon", "asap", "rn", "tho",
                "her", "him", "them", "me", "you", "again", "today", "tomorrow",
                "tonight"}
        first = cand.split()[0].lower()
        if first in skip:
            return None
        return cand.lower()
    return None


def _extract_place(text: str, *, prefer_destination: bool = False) -> str | None:
    """Generic place extractor. Returns the first @/at/to/in/from match."""
    # to-place is preferred for ride/maps/travel destinations
    if prefer_destination:
        m = _RE_TO_PLACE.search(text)
        if m:
            return _clean_place(m.group("place"))
    # airport codes (uber to JFK)
    for tok in re.findall(r"\b([A-Z]{3})\b", text):
        if tok.lower() in _AIRPORTS:
            return tok
    m = _RE_AT_PLACE.search(text)
    if m:
        return _clean_place(m.group("place"))
    m = _RE_IN_PLACE.search(text)
    if m:
        cand = _clean_place(m.group("place"))
        # avoid filler tokens after "in" ("in the meantime", "in fact")
        if cand and cand.lower() not in {"the", "fact", "general", "case"}:
            return cand
    if not prefer_destination:
        m = _RE_TO_PLACE.search(text)
        if m:
            return _clean_place(m.group("place"))
    return None


def _clean_place(raw: str) -> str:
    """Trim trailing fillers like 'for', 'with' that the lookahead missed."""
    s = raw.strip()
    s = re.sub(r"\s+(?:for|with|to|by|on|in|at)\s*$", "", s, flags=re.IGNORECASE)
    return s.rstrip(" ,.;:")


def _extract_bill_kind(text: str) -> str | None:
    text_l = text.lower()
    for kw, canonical in _BILL_KEYWORDS.items():
        if kw in text_l:
            return canonical
    return None


def _extract_food_item(text: str) -> str | None:
    m = _RE_FOOD_HINT.search(text)
    if m:
        return m.group("item").strip().lower()
    return None


def _extract_shopping_item(text: str) -> str | None:
    m = _RE_SHOPPING_HINT.search(text)
    if m:
        item = m.group("item").strip().lower()
        # avoid trapping verbs like "got" when nothing followed
        if len(item) >= 2 and item not in {"it", "one", "two"}:
            return item
    return None


def _extract_qty(text: str) -> str | None:
    m = _RE_QTY.search(text)
    if m:
        return m.group(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-intent fillers
# ─────────────────────────────────────────────────────────────────────────────
def _slots_money(text: str) -> dict[str, Any]:
    amount = _extract_amount(text) or _extract_word_amount(text)
    return {
        "amount":    amount,
        "recipient": _extract_recipient(text),
        "method":    _extract_pay_method(text),
        "note":      _extract_money_note(text),
        "direction": _extract_pay_direction(text),
    }


# Word amounts: "five dollars" → "5", "twenty bucks" → "20"
_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "grand": 1000, "k": 1000,
}

def _extract_word_amount(text: str) -> str | None:
    """Parse written-out amounts: 'five dollars' → '5', 'twenty five bucks' → '25'."""
    text_l = text.lower()
    # Pattern: (word_num) (word_num)? (dollars|bucks|etc)?
    m = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
        r"(?:\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand))?"
        r"(?:\s+(?:dollars?|bucks?|rupees?|pounds?|euros?))?",
        text_l,
    )
    if not m:
        return None
    w1 = _WORD_TO_NUM.get(m.group(1), 0)
    w2 = _WORD_TO_NUM.get(m.group(2), 0) if m.group(2) else 0
    # "twenty five" = 20 + 5; "five hundred" = 5 * 100
    if w1 >= 100 and w2 > 0:
        total = w1 * w2
    elif w2 >= 100:
        total = w1 * w2
    else:
        total = w1 + w2
    return str(total) if total > 0 else None


def _extract_pay_direction(text: str) -> str | None:
    """Detect if the user is sending, requesting, or it's ambiguous."""
    text_l = text.lower()
    # requesting money FROM someone
    req = re.search(
        r"\b(you owe|owe me|pay me|send me|venmo me|cashapp me|zelle me|"
        r"pay (?:me )?back|give me|give me my money|where'?s my money|"
        r"I need .{0,20} from you|lend me)\b",
        text_l,
    )
    if req:
        return "request"
    # sending money TO someone
    send = re.search(
        r"\b(I'?ll pay|I'?ll send|let me pay|let me send|paying|sending|"
        r"transfer(?:ring)? .{0,20} to|I owe|I should pay|"
        r"venmo .{0,20} to|send .{0,20} to|pay .{0,20} to)\b",
        text_l,
    )
    if send:
        return "send"
    return None


def _extract_pay_method(text: str) -> str | None:
    for kw in ("venmo", "cashapp", "cash app", "zelle", "paypal",
               "paytm", "gpay", "phonepe", "upi"):
        if kw in text.lower():
            return kw.replace(" ", "")
    return None


def _extract_money_note(text: str) -> str | None:
    m = re.search(r"\bfor\s+(?P<note>[\w &'\-]{2,40})", text, re.I)
    if m:
        return m.group("note").strip().lower()
    return None


def _slots_alarm(text: str) -> dict[str, Any]:
    when = _parse_when(text)
    note = None
    # everything after "to" or "about" reads as the note
    m = re.search(r"\b(?:to|about|that|for)\s+(?P<note>.{2,80})", text, re.I)
    if m:
        # strip trailing time phrases
        n = re.sub(
            r"\s+(?:at|on|tomorrow|today|tonight|tmrw|next\s+\w+|this\s+\w+|"
            r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*.*$",
            "",
            m.group("note"),
            flags=re.I,
        ).strip(" .,;:")
        if n:
            note = n
    if note is None:
        # last-resort: take the verb phrase preceding the time
        m2 = re.search(r"^(?:remind\s+me\s+(?:to\s+)?|alarm\s+(?:for|to)\s+|"
                       r"set\s+(?:an\s+|a\s+)?alarm\s+(?:for\s+|to\s+)?)"
                       r"(?P<note>[\w &']+)", text.strip(), re.I)
        if m2:
            note = m2.group("note").strip()
    return {
        "datetime":   when,
        "note":       note,
        "recurrence": _extract_recurrence(text),
    }


def _extract_recurrence(text: str) -> str | None:
    text_l = text.lower()
    if re.search(r"\bevery\s+(day|morning|night|monday|tuesday|wednesday|thursday|"
                 r"friday|saturday|sunday|week|month|year)\b", text_l):
        m = re.search(r"\bevery\s+(\w+)", text_l)
        return f"every_{m.group(1)}" if m else "recurring"
    if "daily" in text_l:
        return "daily"
    if "weekly" in text_l:
        return "weekly"
    return None


def _slots_contact(text: str) -> dict[str, Any]:
    return {
        "name":    _extract_contact_name(text),
        "channel": _extract_channel(text) or "call",  # default to call
        "note":    None,
    }


def _slots_calendar(text: str) -> dict[str, Any]:
    when = _parse_when(text)
    title = None
    m = re.search(
        r"\b(meeting|lunch|dinner|coffee|brunch|breakfast|drinks|happy hour|"
        r"interview|standup|sync|1:1|one on one|catch[- ]?up|appointment|call)\b",
        text,
        re.I,
    )
    if m:
        title = m.group(1).lower()
    with_who = _extract_recipient(text) or _extract_contact_name(text)
    if with_who == "__pronoun__":
        with_who = None
    location = _extract_place(text)
    return {
        "title":    title,
        "datetime": when,
        "with_who": with_who,
        "location": location,
    }


def _slots_maps(text: str) -> dict[str, Any]:
    dest = _extract_place(text, prefer_destination=True)
    origin = None
    m = re.search(r"\bfrom\s+(?P<o>[\w&'\- ]{2,40}?)(?=\s+to|[.,]|$)", text, re.I)
    if m:
        origin = m.group("o").strip()
    return {
        "destination": dest,
        "origin":      origin,
        "mode":        _extract_travel_mode(text),
    }


def _extract_travel_mode(text: str) -> str | None:
    text_l = text.lower()
    for mode in ("uber", "lyft", "ola", "rapido", "auto", "rickshaw",
                 "taxi", "walking", "drive", "driving", "transit", "bus", "train"):
        if mode in text_l:
            return mode if mode != "driving" else "drive"
    return None


def _slots_ride(text: str) -> dict[str, Any]:
    return {
        "destination": _extract_place(text, prefer_destination=True),
        "pickup":      None,
        "when":        _parse_when(text),
        "service":     _extract_travel_mode(text),
    }


def _slots_food_order(text: str) -> dict[str, Any]:
    provider = None
    for p in ("doordash", "ubereats", "uber eats", "grubhub", "swiggy", "zomato",
              "deliveroo", "deliver"):
        if p in text.lower():
            provider = p.replace(" ", "")
            break
    return {
        "item":     _extract_food_item(text),
        "provider": provider,
        "when":     _parse_when(text),
        "address":  None,
    }


def _slots_travel(text: str) -> dict[str, Any]:
    return {
        "destination": _extract_place(text, prefer_destination=True),
        "origin":      None,
        "when":        _parse_when(text),
        "mode":        _extract_travel_mode(text)
                       or ("flight" if "fly" in text.lower() or "flight" in text.lower() else None),
    }


def _slots_shopping(text: str) -> dict[str, Any]:
    provider = None
    for p in ("amazon", "flipkart", "ebay", "etsy", "walmart", "target", "bestbuy", "costco"):
        if p in text.lower():
            provider = p
            break
    return {
        "item":     _extract_shopping_item(text),
        "qty":      _extract_qty(text),
        "provider": provider,
        "when":     _parse_when(text),
    }


def _slots_music(text: str) -> dict[str, Any]:
    provider = None
    for p in ("spotify", "apple music", "youtube music", "pandora", "tidal", "amazon music"):
        if p in text.lower():
            provider = p.replace(" ", "_")
            break
    artist = None
    m = re.search(
        r"\bby\s+(?P<a>[A-Z][\w &'\-.]{1,40}?)(?=\s+(?:on|from|via|tonight|today|tomorrow)|[,.]|$)",
        text,
    )
    if m:
        artist = m.group("a").strip()
    song = None
    m2 = re.search(r"\bplay\s+(?P<s>[\w &'\-.]{2,40}?)(?=\s+by\s+|\s+on\s+|[,.]|$)", text, re.I)
    if m2:
        song = m2.group("s").strip()
        if song.lower() in {"music", "something", "anything", "a song"}:
            song = None
    return {"song": song, "artist": artist, "provider": provider}


def _slots_video(text: str) -> dict[str, Any]:
    provider = None
    for p in ("netflix", "hulu", "hbo", "max", "prime video", "disney", "youtube",
              "apple tv", "peacock", "hotstar"):
        if p in text.lower():
            provider = p.replace(" ", "_")
            break
    title = None
    m = re.search(r"\bwatch(?:ing)?\s+(?P<t>[\w &'\-:.]{2,40}?)(?=\s+on\s+|[,.]|$)", text, re.I)
    if m:
        t = m.group("t").strip()
        if t.lower() not in {"tv", "something", "a movie", "a show"}:
            title = t
    return {"title": title, "provider": provider}


def _slots_tickets(text: str) -> dict[str, Any]:
    event = None
    m = re.search(
        r"\btickets?\s+(?:to|for)\s+(?P<e>[\w &'\-:.]{2,60}?)(?=\s+(?:on|at|tomorrow|today)|[,.]|$)",
        text, re.I,
    )
    if m:
        event = m.group("e").strip()
    return {"event": event, "when": _parse_when(text), "qty": _extract_qty(text)}


def _slots_reservation(text: str) -> dict[str, Any]:
    return {
        "place": _extract_place(text),
        "when":  _parse_when(text),
        "qty":   _extract_qty(text),
    }


def _slots_task(text: str) -> dict[str, Any]:
    action = None
    m = re.search(r"\b(?:to|task:?)\s+(?P<a>[\w &'\-]{2,80})", text, re.I)
    if m:
        action = m.group("a").strip()
    return {"action": action, "when": _parse_when(text)}


def _slots_note(text: str) -> dict[str, Any]:
    body = None
    m = re.search(r"\b(?:note|jot down|write down|remember)\b[: ]+(?P<n>.{2,200})", text, re.I)
    if m:
        body = m.group("n").strip()
    return {"body": body or text.strip()}


def _slots_bills(text: str) -> dict[str, Any]:
    return {
        "bill_kind": _extract_bill_kind(text),
        "amount":    _extract_amount(text),
        "due":       _parse_when(text),
    }


def _slots_health(text: str) -> dict[str, Any]:
    provider = None
    for p in ("doctor", "dentist", "physio", "therapist", "physician", "pediatrician",
              "ob/gyn", "ophthalmologist", "dermatologist"):
        if p in text.lower():
            provider = p
            break
    kind = "appointment" if "appointment" in text.lower() else None
    return {"provider": provider, "when": _parse_when(text), "kind": kind}


def _slots_weather(text: str) -> dict[str, Any]:
    place = _extract_place(text)
    return {"place": place, "when": _parse_when(text)}


_FILLERS = {
    "money":       _slots_money,
    "alarm":       _slots_alarm,
    "contact":     _slots_contact,
    "calendar":    _slots_calendar,
    "maps":        _slots_maps,
    "ride":        _slots_ride,
    "food_order":  _slots_food_order,
    "travel":      _slots_travel,
    "shopping":    _slots_shopping,
    "music":       _slots_music,
    "video":       _slots_video,
    "tickets":     _slots_tickets,
    "reservation": _slots_reservation,
    "task":        _slots_task,
    "note":        _slots_note,
    "bills":       _slots_bills,
    "health":      _slots_health,
    "weather":     _slots_weather,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def extract_slots(text: str, intents: list[str]) -> dict[str, dict[str, Any]]:
    """
    Run Layer A regex/dateparser slot extraction for every intent in `intents`.

    Each per-intent dict carries the schema slots plus `_required_filled`
    (bool) and `_missing` (list of slot names whose value is None or pronoun).
    """
    out: dict[str, dict[str, Any]] = {}
    for intent in intents:
        filler = _FILLERS.get(intent)
        if filler is None:
            out[intent] = {"_required_filled": True, "_missing": []}
            continue
        slots = filler(text)
        schema = SLOT_SCHEMA.get(intent, {"required": [], "optional": []})
        missing = []
        for r in schema["required"]:
            v = slots.get(r)
            if v is None or v == "" or v == "__pronoun__":
                missing.append(r)
        slots["_required_filled"] = not missing
        slots["_missing"] = missing
        out[intent] = slots
    return out


def needs_clarification(slots: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """
    Returns list of (intent, missing_slot) pairs that need user clarification.
    Caller can render a chip / clarification card for each.
    """
    pairs = []
    for intent, s in slots.items():
        for missing in s.get("_missing", []):
            pairs.append((intent, missing))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (runs on `python eval/slot_filler.py`)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        ("call dad",                                   ["contact"]),
        ("call him",                                   ["contact"]),
        ("text mom at 6pm",                            ["contact", "alarm"]),
        ("venmo me 20 for pizza",                      ["money"]),
        ("zelle 50 to sarah",                          ["money"]),
        ("uber to JFK at 5am tomorrow",                ["ride", "maps", "alarm"]),
        ("remind me to pay rent friday",               ["alarm", "bills"]),
        ("doordash sushi at 7pm",                      ["food_order"]),
        ("set an alarm for 6am",                       ["alarm"]),
        ("don't forget to call mom thursday",          ["contact", "alarm"]),
        ("flight to Tokyo next month",                 ["travel"]),
        ("book a table at Blue Bottle for 4 friday",   ["reservation"]),
        ("buy headphones on amazon",                   ["shopping"]),
        ("play Anti-Hero by Taylor Swift on spotify",  ["music"]),
        ("watching Stranger Things on netflix",        ["video"]),
        ("dentist appointment monday at 10am",         ["health", "calendar"]),
        ("is it gonna rain in mumbai tomorrow",        ["weather"]),
        ("kal subah 6 baje uber book kar",             ["ride", "alarm"]),
        ("mom ko phone karna hai",                     ["contact"]),
    ]
    print(f"Running {len(cases)} slot-filler self-tests at {_now().isoformat()}\n")
    for text, intents in cases:
        slots = extract_slots(text, intents)
        clar = needs_clarification(slots)
        print(f"  {text!r}")
        for intent, s in slots.items():
            display = {k: v for k, v in s.items() if not k.startswith("_") and v is not None}
            mark = "OK" if s["_required_filled"] else "needs:" + ",".join(s["_missing"])
            print(f"    {intent:<11} [{mark}] {display}")
        if clar:
            print(f"    --> clarification: {clar}")
        print()
