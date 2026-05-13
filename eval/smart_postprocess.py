"""
Smart post-processing — suppresses false co-fires from the model.

The model is good at detecting INDIVIDUAL intents but sometimes co-fires
intents that shouldn't stack. Example: "uber to JFK at 5am" fires ride +
maps + alarm, but the "5am" belongs to the ride, not alarm.

These rules run AFTER model inference and BEFORE returning results.
They're fast (regex, <1ms), deterministic, and only suppress — never add.

Rules:
  1. Suppress alarm when time belongs to another intent (no alarm keywords)
  2. Suppress maps when place is a venue (no navigation keywords)
  3. Suppress calendar on pure reminders (no calendar event keywords)
  4. Suppress reservation on casual "lunch/dinner" (no booking keywords)
"""

import re

# ── Keyword sets ──

_ALARM_KEYWORDS = re.compile(
    r'\b(remind|reminder|alarm|wake|timer|ping me|buzz|poke|alert me|notify|'
    r'set a reminder|set an alarm|don.t let me (forget|sleep|miss)|heads up)\b',
    re.IGNORECASE,
)

_NAVIGATION_KEYWORDS = re.compile(
    r'\b(direction|navigate|heading to|headed to|on my way|omw|'
    r'meet me at|where is|where.s|how do (i|we) get|how far|'
    r'take me to|drive to|driving to|route to|pull up|drop a pin|'
    r'going to|go to|come to|find .* on maps|map me|map to|'
    r'i.m at|im at|waiting at|currently at|parked at)\b',
    re.IGNORECASE,
)

_CALENDAR_EVENT_KEYWORDS = re.compile(
    r'\b(meeting|lunch|dinner|brunch|coffee|drinks|'
    r'standup|sync|1:1|interview|class|lecture|yoga|gym|'
    r'birthday|wedding|graduation|party|concert|game night|'
    r'appointment|schedule|book .* calendar|put .* calendar|'
    r'event|offsite|flight|train)\b',
    re.IGNORECASE,
)

_BOOKING_KEYWORDS = re.compile(
    r'\b(book|reserve|reservation|table for|make a reservation|'
    r'get a table|book us|book a table)\b',
    re.IGNORECASE,
)

# Pattern that suggests a specific venue is mentioned (for reservation Rule 4)
# "at sweetgreen", "at Blue Bottle", "at Nobu" etc.
# Uses negative lookahead to skip time/number words that follow "at".
_VENUE_PATTERN = re.compile(
    r'\bat\s+(?!\d|noon|night|midnight|dawn|dusk|home\b|my\b|the\b|a\b|an\b)[a-z]\w+',
    re.IGNORECASE,
)

# Intents that "own" a time mention (if they fire, alarm shouldn't co-fire)
_TIME_OWNING_INTENTS = {
    'ride', 'travel', 'video', 'tickets', 'food_order',
    'reservation', 'health', 'calendar',
}

# Intents that "own" a place mention (if they fire, maps shouldn't co-fire)
_VENUE_OWNING_INTENTS = {
    'calendar', 'reservation', 'tickets', 'video',
}


def smart_suppress(text: str, fired: list[str], scores: dict[str, float] | None = None) -> list[str]:
    """
    Suppress false co-fires based on keyword analysis.

    Args:
        text: the original message
        fired: list of intent names that the model fired
        scores: optional dict of intent->probability (for logging)

    Returns:
        filtered list of fired intents (only removes, never adds)
    """
    fired_set = set(fired)
    suppress = set()

    # ── Rule 1: Suppress alarm when time belongs to another intent ──
    # "uber at 5am" → 5am is for the ride. alarm only fires if user
    # explicitly asks for reminder/alarm/wake/timer.
    if 'alarm' in fired_set:
        other_time_intents = fired_set & _TIME_OWNING_INTENTS
        if other_time_intents and not _ALARM_KEYWORDS.search(text):
            suppress.add('alarm')

    # ── Rule 2: Suppress maps when place is a venue ──
    # "lunch at sweetgreen friday" → sweetgreen is the venue for lunch.
    # maps only fires if user explicitly asks for navigation/directions.
    if 'maps' in fired_set:
        venue_intents = fired_set & _VENUE_OWNING_INTENTS
        if venue_intents and not _NAVIGATION_KEYWORDS.search(text):
            suppress.add('maps')

    # ── Rule 3: Suppress calendar on pure reminders ──
    # "remind me about the thing tomorrow" → alarm, NOT calendar.
    # calendar only fires if there's an actual event mentioned.
    if 'calendar' in fired_set and 'alarm' in fired_set:
        if _ALARM_KEYWORDS.search(text) and not _CALENDAR_EVENT_KEYWORDS.search(text):
            suppress.add('calendar')

    # ── Rule 4: Suppress reservation on casual meal mentions ──
    # "lunch on may 15th" → calendar, NOT reservation.
    # reservation only fires if user explicitly asks to book/reserve,
    # OR if a specific venue is mentioned ("dinner at Sweetgreen").
    if 'reservation' in fired_set and 'calendar' in fired_set:
        if not _BOOKING_KEYWORDS.search(text) and not _VENUE_PATTERN.search(text):
            suppress.add('reservation')

    return [i for i in fired if i not in suppress]


# ── Quick self-test ──
if __name__ == "__main__":
    tests = [
        ("uber to JFK at 5am tomorrow", ["alarm", "maps", "ride"]),
        ("lunch with sarah friday at 1 at sweetgreen", ["contact", "calendar", "maps"]),
        ("oppenheimer at amc tonight at 9", ["maps", "video", "tickets"]),
        ("lunch on may 15th at noon", ["calendar", "reservation"]),
        ("hey can you remind me about the thing tomorrow", ["alarm", "calendar"]),
        ("remind me at 5am", ["alarm"]),
        ("set alarm for 7am", ["alarm"]),
        ("meet me at sweetgreen", ["maps"]),
        ("directions to sweetgreen", ["maps"]),
        ("book a table at sweetgreen friday at 7", ["reservation", "calendar", "maps"]),
        ("dinner friday at sweetgreen at 7 and catch oppenheimer after", ["calendar", "reservation", "video"]),
        ("uber to JFK at 5am, remind me to pack tonight", ["alarm", "maps", "ride"]),
        ("flight at 6am, set alarm for 5am", ["travel", "alarm"]),
    ]

    print("Smart post-processing self-test:\n")
    for text, fired in tests:
        result = smart_suppress(text, fired)
        suppressed = set(fired) - set(result)
        status = f"  suppressed: {suppressed}" if suppressed else "  (no change)"
        print(f"  {text!r}")
        print(f"    before: {fired}")
        print(f"    after:  {result}{status}")
        print()
