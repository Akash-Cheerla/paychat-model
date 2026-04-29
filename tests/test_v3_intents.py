"""
v3 super-app intent integration test.

Exercises all 18 intents end-to-end through the FastAPI TestClient with a
mocked model (keyword → sigmoid). This catches regressions in:
  - intent extraction (payload field shapes)
  - action_url deep links (one-tap fire)
  - per-intent threshold logic
  - multi-intent firing in a single message

The real model isn't trained on the v3 heads yet, so we mock inference to
synthesize confidence signals from keywords. The extractors + URL builders
are real code under test.

Run:  python tests/test_v3_intents.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as paychat_app  # noqa: E402


# Map keywords (lower) to one or more intents that should fire on that input.
# Matches what we'd expect a well-trained v3 model to do.
_INTENT_KEYWORDS = {
    "money":       ["$", "venmo", "owe", "pay me", "split", "cashapp", "zelle"],
    "alarm":       ["wake me", "set alarm", "set an alarm", "remind me to take"],
    "contact":     ["+1 ", "+91 ", "save this number", "save the number", "save my number"],
    "calendar":    ["meeting", "schedule", "doctor appointment", "team standup", "interview"],
    "maps":        ["directions to", "navigate to", "meet me at", "starbucks"],
    "food_order":  ["doordash", "uber eats", "order pizza", "ordering pizza", "let's order", "lets order", "ordering food", "wings tonight"],
    "ride":        ["uber to", "lyft to", "ola to", "ubering", "lyfting", "call an uber", "calling an uber", "calling uber"],
    "travel":      ["flights to", "trip to", "vacation", "going to tokyo", "hotel in", "airbnb"],
    "shopping":    ["order me", "buy me", "amazon", "ordering a", "i need to order"],
    "music":       ["play ", "spotify", "song by", "listen to", "queue", "playlist"],
    "video":       ["watch ", "netflix", "movie night", "binge", "stream "],
    "tickets":     ["tickets to", "tickets for", "concert", "ticketmaster", "seatgeek", "got tickets"],
    "reservation": ["book a table", "reservation for", "opentable", "table for", "book nobu"],
    "task":        ["i need to", "i have to", "todo:", "still need to", "remember to ", "i should "],
    "note":        ["save this link", "save this for later", "note to self", "bookmarking"],
    "bills":       ["rent due", "rent's due", "pay rent", "electric bill", "phone bill", "credit card bill", "wifi bill"],
    "health":      ["pharmacy", "doctor appointment", "refill", "prescription", "dentist"],
    "weather":     ["weather", "forecast", "is it raining", "is it sunny", "is it cold"],
}


def mock_inference(text: str):
    """Synthesize a 18-head sigmoid vector from keyword presence."""
    t = text.lower()
    intent_probs = {intent: 0.01 for intent in paychat_app.INTENTS}
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(k in t for k in kws):
            intent_probs[intent] = 0.92
    paychat_app.stats["requests"] += 1
    paychat_app.stats["_latency_sum"] += 1.0
    paychat_app.stats["avg_latency_ms"] = (
        paychat_app.stats["_latency_sum"] / paychat_app.stats["requests"]
    )
    return {"intent_probs": intent_probs, "latency_ms": 1.0}


# Bypass real model loading
paychat_app.load_model = lambda *a, **kw: None
paychat_app.model_state["model"] = "MOCK"
paychat_app.model_state["tokenizer"] = "MOCK"
paychat_app.model_state["num_labels"] = len(paychat_app.INTENTS)
paychat_app.model_state["label_order"] = list(paychat_app.INTENTS)
paychat_app.run_inference = mock_inference

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(paychat_app.app)


# ── Tiny test framework ────────────────────────────────────────────────────
PASS = "[ ok ]"
FAIL = "[FAIL]"
results = []


def check(label, cond, detail=""):
    results.append((label, bool(cond), detail))
    status = PASS if cond else FAIL
    suffix = f" -- {detail}" if detail and not cond else ""
    print(f"  {status} {label}{suffix}")


def detect(text, chat_id=None):
    payload = {"text": text}
    if chat_id:
        payload["chat_id"] = chat_id
    r = client.post("/detect", json=payload)
    assert r.status_code == 200, f"/detect failed: {r.status_code} {r.text}"
    return r.json()


def intent_of(resp, intent_type):
    for it in resp.get("intents", []):
        if it["type"] == intent_type:
            return it
    return None


def reset_world():
    paychat_app.popup_tracker.clear()
    if hasattr(paychat_app, "chat_history"):
        paychat_app.chat_history.clear()
    paychat_app.stats.update({
        "requests": 0,
        "money_detected": 0,
        "intents_detected": {i: 0 for i in paychat_app.INTENTS},
        "popups_fired": 0,
        "popups_suppressed": 0,
        "avg_latency_ms": 0.0,
        "_latency_sum": 0.0,
    })


def header(title):
    print(f"\n-- {title} " + "-" * max(2, 60 - len(title)))


# ────────────────────────────────────────────────────────────────────────────
#  Per-intent assertions: payload + action_url shape
# ────────────────────────────────────────────────────────────────────────────

def test_food_order():
    header("food_order")
    reset_world()
    r = detect("let's order pizza on doordash tonight")
    f = intent_of(r, "food_order")
    check("food_order detected", f is not None)
    check("item=pizza", f and f["payload"].get("item") == "pizza")
    check("provider=doordash", f and f["payload"].get("provider_hint") == "doordash")
    check("action_url=doordash", f and "doordash.com" in (f["payload"].get("action_url") or ""))


def test_ride():
    header("ride")
    reset_world()
    r = detect("calling an uber to JFK")
    f = intent_of(r, "ride")
    check("ride detected", f is not None)
    check("dropoff contains JFK", f and "JFK" in (f["payload"].get("dropoff") or ""))
    check("provider=uber", f and f["payload"].get("provider_hint") == "uber")
    check("action_url=uber.com", f and "uber.com" in (f["payload"].get("action_url") or ""))


def test_travel():
    header("travel")
    reset_world()
    r = detect("flights to Tokyo for next month")
    f = intent_of(r, "travel")
    check("travel detected", f is not None)
    check("destination contains Tokyo",
          f and "Tokyo" in (f["payload"].get("destination") or ""))
    check("action_url is travel link",
          f and "google.com/travel" in (f["payload"].get("action_url") or ""))


def test_shopping():
    header("shopping")
    reset_world()
    r = detect("order me 2 phone chargers from amazon")
    f = intent_of(r, "shopping")
    check("shopping detected", f is not None)
    check("item is phone chargers",
          f and f["payload"].get("item") and "phone charger" in f["payload"]["item"].lower())
    check("qty=2", f and f["payload"].get("qty") == "2")
    check("action_url=amazon.com",
          f and "amazon.com" in (f["payload"].get("action_url") or ""))


def test_music():
    header("music")
    reset_world()
    r = detect("play Espresso by Sabrina Carpenter on spotify")
    f = intent_of(r, "music")
    check("music detected", f is not None)
    check("track=Espresso",
          f and (f["payload"].get("track") or "").lower().startswith("espresso"))
    check("artist contains Sabrina",
          f and "Sabrina" in (f["payload"].get("artist") or ""))
    check("action_url=spotify",
          f and "open.spotify.com" in (f["payload"].get("action_url") or ""))


def test_video():
    header("video")
    reset_world()
    r = detect("let's watch Dune Part Two on netflix tonight")
    f = intent_of(r, "video")
    check("video detected", f is not None)
    check("title contains Dune",
          f and "Dune" in (f["payload"].get("title") or ""))
    check("action_url=netflix",
          f and "netflix.com" in (f["payload"].get("action_url") or ""))


def test_tickets():
    header("tickets")
    reset_world()
    r = detect("got tickets to Coldplay on Friday")
    f = intent_of(r, "tickets")
    check("tickets detected", f is not None)
    check("event contains Coldplay",
          f and "Coldplay" in (f["payload"].get("event") or ""))
    check("action_url=ticketmaster",
          f and "ticketmaster.com" in (f["payload"].get("action_url") or ""))


def test_reservation():
    header("reservation")
    reset_world()
    r = detect("book a table at Nobu for 4 people tomorrow")
    f = intent_of(r, "reservation")
    check("reservation detected", f is not None)
    check("venue contains Nobu",
          f and "Nobu" in (f["payload"].get("venue") or ""))
    check("party_size=4", f and f["payload"].get("party_size") == 4)
    check("action_url=opentable",
          f and "opentable.com" in (f["payload"].get("action_url") or ""))


def test_task():
    header("task")
    reset_world()
    r = detect("i need to finish the report by tomorrow")
    f = intent_of(r, "task")
    check("task detected", f is not None)
    check("title contains 'finish the report'",
          f and "finish the report" in (f["payload"].get("title") or "").lower())
    check("action_url is google tasks",
          f and "tasks.google.com" in (f["payload"].get("action_url") or ""))


def test_note():
    header("note")
    reset_world()
    r = detect("save this link: https://example.com/post")
    f = intent_of(r, "note")
    check("note detected", f is not None)
    check("url present",
          f and (f["payload"].get("url") or "").startswith("http"))
    check("action_url=mailto",
          f and (f["payload"].get("action_url") or "").startswith("mailto"))


def test_bills():
    header("bills")
    reset_world()
    r = detect("rent's due on the 1st, $2400")
    f = intent_of(r, "bills")
    check("bills detected", f is not None)
    check("kind=rent", f and f["payload"].get("kind") == "rent")
    check("amount has $2400",
          f and "2400" in (f["payload"].get("amount") or ""))
    check("action_url is google search",
          f and "google.com/search" in (f["payload"].get("action_url") or ""))


def test_health():
    header("health")
    reset_world()
    r = detect("need to refill my prescription, pharmacy near me")
    f = intent_of(r, "health")
    check("health detected", f is not None)
    check("kind=pharmacy", f and f["payload"].get("kind") == "pharmacy")
    check("action_url is google maps",
          f and "google.com/maps" in (f["payload"].get("action_url") or ""))


def test_weather():
    header("weather")
    reset_world()
    r = detect("what's the weather in Tokyo tomorrow")
    f = intent_of(r, "weather")
    check("weather detected", f is not None)
    check("location contains Tokyo",
          f and "Tokyo" in (f["payload"].get("location") or ""))
    check("action_url is google search",
          f and "google.com/search" in (f["payload"].get("action_url") or ""))


# ────────────────────────────────────────────────────────────────────────────
#  Cross-intent multi-fire scenarios (real-world chat)
# ────────────────────────────────────────────────────────────────────────────

def test_multi_food_money():
    header("multi-intent: food + money")
    reset_world()
    # Use both a strong food trigger AND a money trigger
    r = detect("ordering pizza on doordash, venmo me $20", chat_id="chat_combo")
    fired = [i["type"] for i in r.get("intents", [])]
    check("food_order fires", "food_order" in fired)
    check("money fires", "money" in fired, f"got {fired}")


def test_multi_calendar_reservation():
    header("multi-intent: calendar + reservation")
    reset_world()
    # 'meeting' fires calendar; 'book a table' fires reservation
    r = detect("book a table at Nobu for our meeting tomorrow at 7pm")
    fired = [i["type"] for i in r.get("intents", [])]
    check("calendar fires", "calendar" in fired)
    check("reservation fires", "reservation" in fired, f"got {fired}")


def test_multi_health_maps():
    header("multi-intent: health + maps")
    reset_world()
    r = detect("directions to nearest pharmacy please")
    fired = [i["type"] for i in r.get("intents", [])]
    check("health fires", "health" in fired)
    check("maps fires", "maps" in fired, f"got {fired}")


# ────────────────────────────────────────────────────────────────────────────
#  Per-intent threshold smoke
# ────────────────────────────────────────────────────────────────────────────

def test_thresholds_apply():
    header("per-intent thresholds")
    reset_world()
    # Force a signal between the loose-bucket and balanced-bucket thresholds
    # and confirm music fires (its threshold is 0.45) but contact doesn't
    # (its threshold is 0.55, the conservative bucket).
    def low_signal(text: str):
        probs = {i: 0.48 for i in paychat_app.INTENTS}  # > 0.45, < 0.50
        paychat_app.stats["requests"] += 1
        paychat_app.stats["_latency_sum"] += 1.0
        paychat_app.stats["avg_latency_ms"] = paychat_app.stats["_latency_sum"] / paychat_app.stats["requests"]
        return {"intent_probs": probs, "latency_ms": 1.0}

    saved = paychat_app.run_inference
    paychat_app.run_inference = low_signal
    try:
        r = detect("ambiguous mid-confidence message")
        fired = {i["type"] for i in r.get("intents", [])}
        # Loose bucket (0.45) -> fires
        check("music fires at 0.48 (threshold 0.45)", "music" in fired, f"got {fired}")
        check("note  fires at 0.48 (threshold 0.45)", "note"  in fired, f"got {fired}")
        # Balanced bucket (0.50) -> doesn't fire
        check("food_order does NOT fire at 0.48 (threshold 0.50)",
              "food_order" not in fired, f"got {fired}")
        # Conservative bucket (0.55) -> doesn't fire
        check("contact does NOT fire at 0.48 (threshold 0.55)",
              "contact" not in fired, f"got {fired}")
    finally:
        paychat_app.run_inference = saved


# ────────────────────────────────────────────────────────────────────────────
#  Action URL shape sanity (build_action_url called directly)
# ────────────────────────────────────────────────────────────────────────────

def test_action_url_safety():
    header("action_url safety")
    # Empty / partial payloads should never crash and should return None or a URL.
    cases = [
        ("food_order", {}),
        ("ride", {}),
        ("travel", {}),
        ("shopping", {}),
        ("music", {}),
        ("video", {}),
        ("tickets", {}),
        ("reservation", {}),
        ("task", {}),
        ("note", {}),
        ("bills", {}),
        ("health", {}),
        ("weather", {}),
        ("alarm", {"label": "wake up"}),       # missing time_iso
        ("contact", {}),                       # missing phone
        ("maps", {"place": None}),
    ]
    for intent, payload in cases:
        try:
            url = paychat_app.build_action_url(intent, payload)
            ok = (url is None) or isinstance(url, str)
            check(f"{intent} empty payload doesn't crash", ok,
                  f"got {url!r}")
        except Exception as e:
            check(f"{intent} empty payload doesn't crash", False, f"raised {e!r}")


# ────────────────────────────────────────────────────────────────────────────
#  Run
# ────────────────────────────────────────────────────────────────────────────

def run():
    test_food_order()
    test_ride()
    test_travel()
    test_shopping()
    test_music()
    test_video()
    test_tickets()
    test_reservation()
    test_task()
    test_note()
    test_bills()
    test_health()
    test_weather()
    test_multi_food_money()
    test_multi_calendar_reservation()
    test_multi_health_maps()
    test_thresholds_apply()
    test_action_url_safety()

    passed = sum(1 for (_, ok, _) in results if ok)
    failed = len(results) - passed
    print()
    print("=" * 70)
    print(f"  {passed}/{len(results)} checks passed, {failed} failed")
    print("=" * 70)
    if failed:
        print("\nFailed checks:")
        for label, ok, detail in results:
            if not ok:
                print(f"  - {label}{(' — ' + detail) if detail else ''}")
        sys.exit(1)


if __name__ == "__main__":
    run()
