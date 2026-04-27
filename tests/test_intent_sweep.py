"""
Comprehensive intent sweep: edge cases for every intent.

Hits the live model + extractor via FastAPI TestClient lifespan, prints PASS/FAIL
for each case, and exits non-zero if anything regressed.

Run:  python tests/test_intent_sweep.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app import app


# Each case: (intent, text, expected_dict_or_predicate)
# expected_dict: keys must be present in payload[intent], values use these markers:
#   - "EXISTS"             -> just must be non-null/non-empty
#   - "MISSING"            -> must be None or empty
#   - "ISO:YYYY-MM-DD"     -> ISO date prefix match
#   - "ISO:YYYY-MM-DDTHH:MM" -> exact ISO match
#   - exact string         -> equality
#   - ("regex", pattern)   -> regex match

CASES = [
    # --------------------------- MONEY ---------------------------
    # Money payload schema: {amount: str|None, trigger_type: str, direction: str}
    ("money", "send john 50 dollars",
        {"amount": ("regex", r"50"), "trigger_type": "direct_amount"}),
    ("money", "request $25 from Sarah for pizza",
        {"amount": ("regex", r"25"), "trigger_type": "direct_amount"}),
    ("money", "venmo Akash 12.50",
        {"trigger_type": "payment_app"}),
    ("money", "send 10 bucks to mom",
        {"amount": ("regex", r"10"), "trigger_type": "direct_amount"}),
    ("money", "split the bill",
        {"trigger_type": "EXISTS"}),
    ("money", "I owe you $40",
        {"amount": ("regex", r"40")}),

    # --------------------------- ALARM ---------------------------
    ("alarm", "wake me up at 6am tomorrow", {"time_iso": ("regex", r"T06:00$")}),
    ("alarm", "set an alarm for 7:30 am", {"time_iso": ("regex", r"T07:30$")}),
    ("alarm", "alarm at 10:45 am", {"time_iso": ("regex", r"T10:45$")}),
    ("alarm", "remind me to take medicine at 9 pm", {"time_iso": ("regex", r"T21:00$")}),
    ("alarm", "wake me up at 5 in the morning", {"time_iso": ("regex", r"T05:00$")}),
    ("alarm", "alarm for 11:15", {"time_iso": "EXISTS"}),
    ("alarm", "wake me up at midnight", {"time_iso": ("regex", r"T00:00$")}),
    ("alarm", "set alarm for noon", {"time_iso": ("regex", r"T12:00$")}),

    # --------------------------- CONTACT -------------------------
    ("contact", "Save the number +1 454 555 1234 under the name test contact",
        {"phone": "EXISTS", "name_hint": "test contact"}),
    ("contact", "save the plumbers number +91 9999999888",
        {"phone": "EXISTS", "name_hint": "plumber"}),
    ("contact", "hey akash save this number +1 415-555-1234",
        {"phone": "EXISTS", "name_hint": "akash"}),
    ("contact", "add John Smith to my contacts, his number is 9876543210",
        {"phone": "EXISTS", "name_hint": ("regex", r"(?i)john")}),
    ("contact", "save +91 9999988888 as Mom",
        {"phone": "EXISTS", "name_hint": ("regex", r"(?i)mom")}),
    ("contact", "save this number +91 8765432109",
        {"phone": "EXISTS"}),  # no name — should not invent one
    ("contact", "add this number +1 415-555-1212 to my contacts",
        {"phone": "EXISTS"}),
    ("contact", "save the number 415-555-9999 named Sarah",
        {"phone": "EXISTS", "name_hint": ("regex", r"(?i)sarah")}),

    # --------------------------- CALENDAR ------------------------
    ("calendar", "Mark the event on the calendar for 28th April as a meeting",
        {"title": ("regex", r"(?i)meeting"), "start_iso": ("regex", r"^2026-04-28")}),
    ("calendar", "I would like to set up a meeting for the 27th at 10 a.m",
        {"title": ("regex", r"(?i)meeting"), "start_iso": ("regex", r"T10:00$")}),
    ("calendar", "schedule lunch with Sarah on Friday at 1pm",
        {"title": "EXISTS", "start_iso": ("regex", r"T13:00$")}),
    ("calendar", "doctor appointment tomorrow at 3:30 pm",
        {"title": "EXISTS", "start_iso": ("regex", r"T15:30$")}),
    ("calendar", "team standup every monday at 9am",
        {"title": "EXISTS", "start_iso": ("regex", r"T09:00$")}),
    # ("remind me ..." trips alarm by classifier, not calendar — covered by alarm cases)
    ("calendar", "interview at 4:15 pm next thursday",
        {"title": "EXISTS", "start_iso": ("regex", r"T16:15$")}),
    ("calendar", "block off 2 hours for deep work tomorrow morning",
        {"title": "EXISTS", "start_iso": "EXISTS"}),

    # --------------------------- MAPS ----------------------------
    ("maps", "Meet me at 216 Vakil Garden City for said meeting",
        {"place": ("regex", r"(?i)216 vakil garden city")}),
    ("maps", "directions to Starbucks on 5th avenue",
        {"place": ("regex", r"(?i)starbucks")}),
    ("maps", "navigate to JFK airport",
        {"place": ("regex", r"(?i)jfk")}),
    ("maps", "where is the nearest pharmacy",
        {"place": "EXISTS"}),
    ("maps", "take me to Indiranagar",
        {"place": ("regex", r"(?i)indiranagar")}),
    ("maps", "show me coffee shops near Brigade Road",
        {"place": "EXISTS"}),
    ("maps", "drive to 1600 Amphitheatre Parkway tomorrow",
        {"place": ("regex", r"(?i)1600 amphitheatre")}),
]


def check(actual, expected_spec):
    """Return (ok, reason). actual is the raw payload value."""
    if expected_spec == "EXISTS":
        ok = actual not in (None, "", [], {})
        return ok, "" if ok else f"expected non-empty, got {actual!r}"
    if expected_spec == "MISSING":
        ok = actual in (None, "", [], {})
        return ok, "" if ok else f"expected empty, got {actual!r}"
    if isinstance(expected_spec, tuple) and expected_spec[0] == "regex":
        if actual is None:
            return False, f"expected match {expected_spec[1]!r}, got None"
        ok = bool(re.search(expected_spec[1], str(actual)))
        return ok, "" if ok else f"expected /{expected_spec[1]}/, got {actual!r}"
    # exact equality
    ok = actual == expected_spec
    return ok, "" if ok else f"expected {expected_spec!r}, got {actual!r}"


def run():
    passed = 0
    failed = 0
    failures = []

    with TestClient(app) as client:
        for intent, text, expected in CASES:
            r = client.post("/detect", json={"text": text})
            if r.status_code != 200:
                failed += 1
                failures.append((intent, text, f"HTTP {r.status_code}"))
                print(f"[FAIL] {intent:9s} | {text!r} -> http {r.status_code}")
                continue

            body = r.json()
            intents_arr = body.get("intents") or []
            match = next((i for i in intents_arr if i.get("type") == intent), None)
            if match is None:
                detected_types = [i.get("type") for i in intents_arr]
                failed += 1
                failures.append((intent, text, f"intent not detected (got {detected_types})"))
                print(f"[FAIL] {intent:9s} | {text!r} -> not detected (got {detected_types})")
                continue
            payload = match.get("payload") or {}

            case_ok = True
            problems = []
            for key, spec in expected.items():
                ok, why = check(payload.get(key), spec)
                if not ok:
                    case_ok = False
                    problems.append(f"{key}: {why}")

            if case_ok:
                passed += 1
                # compact one-line success report
                preview = {k: payload.get(k) for k in expected.keys()}
                print(f"[ ok ] {intent:9s} | {text!r}")
                print(f"         -> {preview}")
            else:
                failed += 1
                failures.append((intent, text, "; ".join(problems)))
                print(f"[FAIL] {intent:9s} | {text!r}")
                for p in problems:
                    print(f"         !! {p}")
                print(f"         payload={payload}")

    print()
    print("=" * 70)
    print(f"  {passed}/{passed + failed} cases passed, {failed} failed")
    print("=" * 70)

    if failures:
        print()
        print("Failures:")
        for intent, text, why in failures:
            print(f"  - [{intent}] {text!r}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    run()
