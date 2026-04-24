"""
End-to-end tests for PayChat multi-intent popup cooldown behavior.

v2.0.0 — covers:
  - Money back-compat flat fields (the original 20 scenarios still pass)
  - Per-intent cooldowns (alarm cooldown doesn't block money, etc.)
  - Multi-intent messages (e.g. "meeting at 3pm, venmo me $20")
  - Intent-scoped /popup-dismissed, /reset-cooldown, /payment-complete
  - intents[] array shape + per-intent payloads (amount, time_iso, phone, place)
  - Targeting signals (addressee / third_party / is_self / is_mutual)

Runs without loading the real DistilBERT model — we mock run_inference so the
cooldown + extractor logic are what get exercised. Time is fast-forwarded by
mutating tracker timestamps directly; the whole suite runs in <2 seconds.

Run:  python tests/test_cooldown.py
"""
import os
import re
import sys

# Make sibling modules importable when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as paychat_app  # noqa: E402


# ── Mock inference ──────────────────────────────────────────────────────────
# The real run_inference returns {"intent_probs": {money, alarm, contact, calendar, maps},
# "latency_ms": X}. We synthesize intent_probs from keywords so the cooldown logic
# + per-intent extractors get a realistic multi-intent input.

_INTENT_KEYWORDS = {
    "money": [
        "$", " money", "venmo", "cashapp", "zelle", "owe", "pay me", "pay up",
        "split", "bucks", "dollar", "send me", "paypal",
    ],
    "alarm": [
        "remind me", "reminder", "wake me", "wake up", "set alarm", "set an alarm",
        "ping me in", "in 20 min", "in 2 hour", "notify me", "alert me",
        "don't let me forget", "note to self",
    ],
    "contact": [
        "save this number", "save his number", "save her number", "save my number",
        "save contact", "add to contacts", "add contact", "here's my number",
        "heres my number", "call me at", "here's the number", "this is ", "my number",
        "number is", "+1 ", "+91 ",
    ],
    "calendar": [
        "meeting", "team sync", "standup", "1:1", "dinner ", "lunch ", "coffee ",
        "schedule", "block off", "pencil me in", "save the date", "wedding",
        "gym ",
    ],
    "maps": [
        "meet me at ", "meet at ", "meet you at ", "i'm at ", "im at ", "currently at",
        "heading to ", "omw to ", "directions to ", "navigate to ", "pull up ",
        "the address is", "address for",
    ],
}


def mock_inference(text: str):
    """Simulate 5-head sigmoid output from keyword matching."""
    t = text.lower()
    intent_probs = {intent: 0.01 for intent in paychat_app.INTENTS}
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(k in t for k in kws):
            intent_probs[intent] = 0.99

    # Stats updates mirror the real run_inference (orchestrator does the rest)
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
paychat_app.model_state["num_labels"] = 5
paychat_app.model_state["label_order"] = list(paychat_app.INTENTS)
paychat_app.run_inference = mock_inference

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(paychat_app.app)


# -- Tiny test framework ----------------------------------------------------
PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  {status} {name}" + (f"   -- {detail}" if detail and not cond else ""))
    results.append((name, bool(cond), detail))


def header(n, title):
    print(f"\n-- Scenario {n}: {title} " + "-" * 40)


def reset_world():
    """Clear all state between scenarios."""
    paychat_app.popup_tracker.clear()
    paychat_app.stats.update({
        "requests": 0,
        "money_detected": 0,
        "intents_detected": {i: 0 for i in paychat_app.INTENTS},
        "popups_fired": 0,
        "popups_suppressed": 0,
        "avg_latency_ms": 0.0,
        "_latency_sum": 0.0,
    })


def detect(text, chat_id=None, sender="akash"):
    payload = {"text": text, "sender": sender}
    if chat_id:
        payload["chat_id"] = chat_id
    r = client.post("/detect", json=payload)
    assert r.status_code == 200, f"/detect failed: {r.status_code} {r.text}"
    return r.json()


def rewind_ts(chat_id, seconds, intent="money"):
    """Simulate `seconds` of wall-clock passing by backdating a tracker entry."""
    key = (chat_id, intent)
    e = paychat_app.popup_tracker.get(key)
    if not e:
        return
    e["last_popup_ts"] -= seconds
    e["last_event_ts"] -= seconds


def tracker(chat_id, intent):
    """Convenience accessor for a (chat_id, intent) tracker entry."""
    return paychat_app.popup_tracker.get((chat_id, intent))


def intent_of(resp, intent_type):
    """Pull a specific intent dict from the intents[] array."""
    for it in resp.get("intents", []):
        if it["type"] == intent_type:
            return it
    return None


# ════════════════════════════════════════════════════════════════════════════
#  MONEY SCENARIOS (back-compat — the original 20 still pass)
# ════════════════════════════════════════════════════════════════════════════

def scenario_1_first_money_fires():
    header(1, "First money message fires the popup")
    reset_world()
    r = detect("yo you still owe me $40 from the airbnb", "chat_A")
    check("is_money=True", r["is_money"] is True)
    check("should_popup=True", r["should_popup"] is True)
    check("suppressed_reason=None", r["suppressed_reason"] is None)
    check("chat_state=cooldown", r["chat_state"] == "cooldown",
          f"got {r['chat_state']}")
    check("detected_amount=$40", r["detected_amount"] == "$40")
    check("direction=request", r["direction"] == "request")
    check("trigger_type=owing_debt", r["trigger_type"] == "owing_debt")
    check("tracker entry created",
          tracker("chat_A", "money") is not None)
    check("popup_count=1",
          tracker("chat_A", "money")["popup_count"] == 1)
    # New contract: intents[] must have money
    mi = intent_of(r, "money")
    check("intents[] contains money", mi is not None)
    check("money intent should_popup=True", mi and mi["should_popup"] is True)
    check("money payload has amount=$40",
          mi and mi["payload"].get("amount") == "$40")


def scenario_2_cooldown_suppresses_follow_ups():
    header(2, "Follow-up nagging is suppressed (cooldown_active)")
    reset_world()
    detect("you still owe me $40", "chat_A")  # initial popup
    for msg in ["fr pay up lol", "bro pay me back now", "venmo me whenever"]:
        r = detect(msg, "chat_A")
        check(f"'{msg}' -> is_money=True", r["is_money"] is True)
        check(f"'{msg}' -> should_popup=False", r["should_popup"] is False,
              f"got should_popup={r['should_popup']}")
        check(f"'{msg}' -> reason=cooldown_active",
              r["suppressed_reason"] == "cooldown_active",
              f"got {r['suppressed_reason']}")
        check(f"'{msg}' -> remaining > 0",
              r["cooldown_remaining_seconds"] > 0)
    t = tracker("chat_A", "money")
    check("popup_count still 1", t["popup_count"] == 1)
    check("suppression_count == 3", t["suppression_count"] == 3)


def scenario_3_non_money_doesnt_affect_cooldown():
    header(3, "Non-money chatter is ignored, doesn't touch cooldown")
    reset_world()
    detect("you owe me $40", "chat_A")
    before = dict(tracker("chat_A", "money"))
    r = detect("yeah yeah my bad lemme grab my phone", "chat_A")
    check("is_money=False on non-money", r["is_money"] is False)
    check("should_popup=False", r["should_popup"] is False)
    check("reason=not_money", r["suppressed_reason"] == "not_money")
    after = tracker("chat_A", "money")
    check("tracker unchanged by non-money msg",
          before["last_popup_ts"] == after["last_popup_ts"]
          and before["popup_count"] == after["popup_count"])
    # intents[] should be empty (nothing fired above threshold)
    check("intents[] empty for non-money chatter",
          r.get("intents") == [])


def scenario_4_new_amount_overrides_cooldown():
    header(4, "Distinct new $ amount in same chat overrides cooldown")
    reset_world()
    detect("you owe me $40", "chat_A")
    rewind_ts("chat_A", 30)  # 30s later — well inside cooldown
    r = detect("oh also you owe me $15 for gas", "chat_A")
    check("should_popup=True on new amount", r["should_popup"] is True,
          f"got should_popup={r['should_popup']}, reason={r.get('suppressed_reason')}")
    check("detected_amount=$15", r["detected_amount"] == "$15")
    t = tracker("chat_A", "money")
    check("popup_count=2", t["popup_count"] == 2)
    check("last_payload.amount=$15",
          t["last_payload"]["amount"] == "$15")


def scenario_5_same_amount_stays_suppressed():
    header(5, "Same $ amount inside cooldown stays suppressed")
    reset_world()
    detect("you owe me $40", "chat_A")
    rewind_ts("chat_A", 30)
    r = detect("bro the $40 pls", "chat_A")
    check("should_popup=False (same amount)", r["should_popup"] is False)
    check("reason=cooldown_active",
          r["suppressed_reason"] == "cooldown_active")


def scenario_6_cooldown_expires_naturally():
    header(6, "Cooldown expires after POPUP_COOLDOWN_SECONDS")
    reset_world()
    detect("you owe me $40", "chat_A")
    rewind_ts("chat_A", paychat_app.POPUP_COOLDOWN_SECONDS + 10)
    r = detect("pay up", "chat_A")
    check("should_popup=True after cooldown expires",
          r["should_popup"] is True,
          f"got reason={r.get('suppressed_reason')}")
    check("popup_count=2",
          tracker("chat_A", "money")["popup_count"] == 2)


def scenario_7_payment_complete_clears_cooldown():
    header(7, "Payment complete clears cooldown -> post_payment grace window")
    reset_world()
    detect("you owe me $40", "chat_A")
    r = client.post("/payment-complete/chat_A",
                    json={"amount": "$40", "payer": "rohit", "method": "venmo"})
    check("/payment-complete returns 200", r.status_code == 200)
    body = r.json()
    check("state=post_payment", body["chat_state"] == "post_payment")
    check("previous_state=cooldown", body["previous_state"] == "cooldown")
    check("response is money-scoped", body.get("intent") == "money")

    # Victory-lap message during grace
    r = detect("sent!", "chat_A")
    check("non-money confirmation -> not_money",
          r["suppressed_reason"] == "not_money")

    # Imagine model misfires on a confirmation; should still suppress
    r = detect("paid you back $40 bro", "chat_A")
    if r["is_money"]:
        check("money msg during grace -> post_payment_grace",
              r["suppressed_reason"] == "post_payment_grace",
              f"got {r['suppressed_reason']}")
        check("should_popup=False during grace", r["should_popup"] is False)


def scenario_8_after_grace_new_topic_pops():
    header(8, "After grace expires, a genuinely new topic pops again")
    reset_world()
    detect("you owe me $40", "chat_A")
    client.post("/payment-complete/chat_A", json={"amount": "$40"})
    # Fast-forward past the 60s grace
    tracker("chat_A", "money")["last_event_ts"] -= (
        paychat_app.POST_PAYMENT_GRACE_SECONDS + 5
    )
    r = detect("also you owe me $15 for gas", "chat_A")
    check("should_popup=True after grace", r["should_popup"] is True,
          f"got reason={r.get('suppressed_reason')}")
    check("popup_count=2",
          tracker("chat_A", "money")["popup_count"] == 2)


def scenario_9_dismissed_has_longer_cooldown():
    header(9, "User dismissed -> 15-min cooldown (vs 5-min default)")
    reset_world()
    detect("you owe me $40", "chat_A")
    r = client.post("/popup-dismissed/chat_A")  # defaults to intent=money
    check("/popup-dismissed returns 200", r.status_code == 200)
    body = r.json()
    check("intent=money (default)", body["intent"] == "money")
    check("state=dismissed", body["chat_state"] == "dismissed")
    check("cooldown_seconds=900", body["cooldown_seconds"] == 900)

    r = detect("bro the money", "chat_A")
    check("should_popup=False (dismissed)", r["should_popup"] is False)
    check("reason=recently_dismissed",
          r["suppressed_reason"] == "recently_dismissed",
          f"got {r['suppressed_reason']}")
    check("remaining > 300 (dismissed uses 900s window)",
          r["cooldown_remaining_seconds"] > 300)


def scenario_10_dismissed_holds_past_normal_cooldown():
    header(10, "Dismissed state holds even after normal 5-min window passes")
    reset_world()
    detect("you owe me $40", "chat_A")
    client.post("/popup-dismissed/chat_A")
    rewind_ts("chat_A", paychat_app.POPUP_COOLDOWN_SECONDS + 30)
    r = detect("pay me", "chat_A")
    check("still suppressed past normal cooldown",
          r["should_popup"] is False,
          f"got should_popup={r['should_popup']}, reason={r.get('suppressed_reason')}")
    check("reason still recently_dismissed",
          r["suppressed_reason"] == "recently_dismissed")


def scenario_11_chats_are_isolated():
    header(11, "Cooldowns isolated per chat_id -- no leakage")
    reset_world()
    detect("you owe me $40", "chat_A")
    r = detect("hey split ubers this weekend?", "chat_B")
    check("chat_B pops even though chat_A in cooldown",
          r["should_popup"] is True,
          f"got reason={r.get('suppressed_reason')}")
    check("chat_B trigger=bill_splitting",
          r["trigger_type"] == "bill_splitting")
    check("two independent tracker entries",
          tracker("chat_A", "money") is not None
          and tracker("chat_B", "money") is not None)
    check("chat_A still in cooldown",
          tracker("chat_A", "money")["state"] == "cooldown")


def scenario_12_no_chat_id_lets_through():
    header(12, "No chat_id -> popup fires (untracked fallback)")
    reset_world()
    r = detect("you owe me $40")  # no chat_id
    check("should_popup=True without chat_id", r["should_popup"] is True)
    check("chat_state=untracked", r["chat_state"] == "untracked",
          f"got {r['chat_state']}")
    check("tracker empty", len(paychat_app.popup_tracker) == 0)


def scenario_13_reset_cooldown_clears():
    header(13, "POST /reset-cooldown/{chat_id} clears tracker")
    reset_world()
    detect("you owe me $40", "chat_A")
    check("entry exists", tracker("chat_A", "money") is not None)
    # No intent arg -> clears everything for that chat
    r = client.post("/reset-cooldown/chat_A")
    check("reset returns 200", r.status_code == 200)
    body = r.json()
    check("cleared_intents includes money",
          "money" in body.get("cleared_intents", []))
    check("entry removed", tracker("chat_A", "money") is None)
    r = detect("you owe me $40 bro", "chat_A")
    check("pops again after reset", r["should_popup"] is True)


def scenario_14_chat_state_endpoint():
    header(14, "GET /chat-state/{chat_id} reports accurate per-intent state")
    reset_world()
    r = client.get("/chat-state/unknown_chat").json()
    check("unknown chat -> state=idle", r["state"] == "idle")
    check("unknown chat has message field", "message" in r)
    check("unknown chat intents dict empty", r.get("intents") == {})

    detect("you owe me $40", "chat_A")
    r = client.get("/chat-state/chat_A").json()
    check("chat-state returns intents dict", "intents" in r)
    check("money intent tracked", "money" in r["intents"])
    mi = r["intents"]["money"]
    check("money state=cooldown", mi["state"] == "cooldown")
    check("money popup_count=1", mi["popup_count"] == 1)
    check("money last_payload.amount=$40",
          mi["last_payload"]["amount"] == "$40")
    check("money cooldown_remaining ~300",
          250 <= mi["cooldown_remaining_seconds"] <= 300,
          f"got {mi['cooldown_remaining_seconds']}")


def scenario_15_metrics_track_accurately():
    header(15, "/metrics counters reflect fire/suppression correctly")
    reset_world()
    detect("you owe me $40", "chat_A")              # fire (money)
    detect("pay up bro", "chat_A")                  # suppress (money cooldown)
    detect("venmo me fr", "chat_A")                 # suppress (money cooldown)
    detect("meeting tomorrow at noon", "chat_A")    # fire (calendar — different intent)
    detect("hey split ubers?", "chat_B")            # fire (money — different chat)

    r = client.get("/metrics").json()
    # 3 money fires, 2 money suppresses (lunch is a different intent — fires once too)
    # So popups_fired = 3 (money: chat_A#1, chat_B#1, calendar: chat_A#1)
    # popups_suppressed = 2 (money: chat_A#2, chat_A#3)
    check("popups_fired=3", r["popups_fired"] == 3,
          f"got {r['popups_fired']}")
    check("popups_suppressed=2", r["popups_suppressed"] == 2,
          f"got {r['popups_suppressed']}")
    check("active_chat_trackers=3", r["active_chat_trackers"] == 3,
          f"got {r['active_chat_trackers']}")
    # money_detected = 4 (detect calls where money probs fired: 3 in chat_A + 1 in chat_B)
    check("money_detected=4", r["money_detected"] == 4,
          f"got {r['money_detected']}")
    check("intents_detected dict has all 5 keys",
          set(r["intents_detected"].keys()) == set(paychat_app.INTENTS))
    check("intents_detected.money=4", r["intents_detected"]["money"] == 4)
    check("intents_detected.calendar>=1",
          r["intents_detected"]["calendar"] >= 1,
          f"got {r['intents_detected']['calendar']}")


def scenario_16_eviction_of_stale_entries():
    header(16, "Stale tracker entries evicted after TRACKER_EVICTION_SECONDS")
    reset_world()
    detect("you owe me $40", "old_chat")
    detect("you owe me $50", "new_chat")
    tracker("old_chat", "money")["last_event_ts"] -= (
        paychat_app.TRACKER_EVICTION_SECONDS + 60
    )
    detect("random non-money", "new_chat")  # triggers opportunistic cleanup
    check("stale entry evicted",
          tracker("old_chat", "money") is None)
    check("fresh entry kept",
          tracker("new_chat", "money") is not None)


def scenario_17_websocket_same_contract():
    header(17, "WebSocket returns the same popup fields as /detect")
    reset_world()
    with client.websocket_connect("/ws/detect") as ws:
        ws.send_json({"text": "you owe me $40", "chat_id": "chat_WS", "sender": "akash"})
        msg = ws.receive_json()
        vd = msg.get("venmo_detection", {})
        check("ws returns venmo_detection", bool(vd))
        check("ws is_money=True", vd.get("is_money") is True)
        check("ws should_popup=True", vd.get("should_popup") is True)
        check("ws chat_state=cooldown", vd.get("chat_state") == "cooldown")
        check("ws passes through chat_id",
              msg.get("chat_id") == "chat_WS")
        # Intents array should come through too
        check("ws includes intents[]",
              isinstance(msg.get("intents"), list) and len(msg["intents"]) >= 1)

        ws.send_json({"text": "pay up", "chat_id": "chat_WS"})
        msg2 = ws.receive_json()
        vd2 = msg2["venmo_detection"]
        check("ws follow-up suppressed",
              vd2["should_popup"] is False
              and vd2["suppressed_reason"] == "cooldown_active")


def scenario_18_empty_text_rejected():
    header(18, "Empty text -> 400")
    r = client.post("/detect", json={"text": "", "chat_id": "chat_A"})
    check("empty text -> 400", r.status_code == 400)
    r = client.post("/detect", json={"text": "   ", "chat_id": "chat_A"})
    check("whitespace-only -> 400", r.status_code == 400)


def scenario_19_payment_complete_on_unknown_chat():
    header(19, "Payment-complete on chat with no tracker entry still works")
    reset_world()
    r = client.post("/payment-complete/brand_new_chat",
                    json={"amount": "$20", "method": "venmo"})
    check("returns 200", r.status_code == 200)
    body = r.json()
    check("state=post_payment", body["chat_state"] == "post_payment")
    check("previous_state=idle", body["previous_state"] == "idle")


def scenario_20_full_conversation_trace():
    header(20, "Full 10-event conversation -- the example from the docs")
    reset_world()

    # 1. First money msg
    r = detect("yo you still owe me $40 from the airbnb", "trip_chat")
    check("#1 popup fires", r["should_popup"] is True)

    # 2-4. Three nags
    for m in ["fr pay up", "bro it's been 2 weeks", "venmo me whenever"]:
        r = detect(m, "trip_chat")
        check(f"#nag '{m}' suppressed", r["should_popup"] is False)

    # 5. Non-money
    r = detect("yeah yeah my bad lemme grab my phone", "trip_chat")
    check("#5 non-money", r["suppressed_reason"] == "not_money")

    # 6. Payment completes
    r = client.post("/payment-complete/trip_chat",
                    json={"amount": "$40", "payer": "rohit", "method": "venmo"})
    check("#6 payment complete OK", r.status_code == 200)

    # 7-8. Confirmation messages during grace
    r = detect("sent!", "trip_chat")
    check("#7 'sent!' not_money", r["suppressed_reason"] == "not_money")
    r = detect("got it thanks bro", "trip_chat")
    check("#8 'thanks' not_money", r["suppressed_reason"] == "not_money")

    # 9. Simulate 45 min passing -> new topic in same chat
    e = tracker("trip_chat", "money")
    e["last_event_ts"] -= 2700
    e["last_popup_ts"] -= 2700
    r = detect("oh also u owe me $15 for gas", "trip_chat")
    check("#9 new amount after 45min pops", r["should_popup"] is True,
          f"reason={r.get('suppressed_reason')}")

    # 10. User dismisses, then more nags are suppressed harder
    client.post("/popup-dismissed/trip_chat")  # defaults to money
    r = detect("bro the gas money", "trip_chat")
    check("#10 post-dismiss suppressed",
          r["suppressed_reason"] == "recently_dismissed")

    m = client.get("/metrics").json()
    check("2 popups fired across conversation", m["popups_fired"] == 2,
          f"got {m['popups_fired']}")


# ════════════════════════════════════════════════════════════════════════════
#  MULTI-INTENT SCENARIOS (the new v2 surface)
# ════════════════════════════════════════════════════════════════════════════

def scenario_21_per_intent_cooldown_isolation():
    header(21, "Alarm cooldown doesn't block a money popup in the same chat")
    reset_world()
    # Fire an alarm
    r = detect("remind me to take meds at 10pm", "chat_X")
    ai = intent_of(r, "alarm")
    check("alarm fires", ai and ai["should_popup"] is True)
    check("alarm tracker created", tracker("chat_X", "alarm") is not None)
    check("money tracker NOT created", tracker("chat_X", "money") is None)

    # Now money in the same chat, immediately
    r = detect("btw you owe me $40 for groceries", "chat_X")
    mi = intent_of(r, "money")
    check("money still fires (different intent)",
          mi and mi["should_popup"] is True,
          f"got {mi and mi['suppressed_reason']}")
    check("money tracker now exists", tracker("chat_X", "money") is not None)
    check("alarm tracker unaffected",
          tracker("chat_X", "alarm")["popup_count"] == 1)


def scenario_22_multi_intent_message_fires_both():
    header(22, "A message with money+alarm fires both popups")
    reset_world()
    r = detect("remind me to venmo priya $25 at 8pm", "chat_Y")
    mi = intent_of(r, "money")
    ai = intent_of(r, "alarm")
    check("money intent present", mi is not None)
    check("alarm intent present", ai is not None)
    check("money should_popup=True", mi and mi["should_popup"] is True)
    check("alarm should_popup=True", ai and ai["should_popup"] is True)
    check("money payload has $25",
          mi and mi["payload"].get("amount") == "$25")
    check("alarm payload has time_iso",
          ai and ai["payload"].get("time_iso") is not None)
    # Both counters ticked
    m = client.get("/metrics").json()
    check("popups_fired=2 for multi-intent", m["popups_fired"] == 2,
          f"got {m['popups_fired']}")


def scenario_23_payment_complete_scoped_to_money():
    header(23, "/payment-complete clears money only, leaves alarm cooldown intact")
    reset_world()
    # Fire both alarm and money in the same chat
    detect("remind me to pay priya at 8pm $25", "chat_Z")
    check("alarm cooldown active",
          tracker("chat_Z", "alarm") is not None
          and tracker("chat_Z", "alarm")["state"] == "cooldown")
    check("money cooldown active",
          tracker("chat_Z", "money") is not None
          and tracker("chat_Z", "money")["state"] == "cooldown")

    client.post("/payment-complete/chat_Z", json={"amount": "$25"})
    check("money state flipped to post_payment",
          tracker("chat_Z", "money")["state"] == "post_payment")
    check("alarm state untouched (still cooldown)",
          tracker("chat_Z", "alarm")["state"] == "cooldown")


def scenario_24_dismiss_scoped_to_intent():
    header(24, "/popup-dismissed?intent=alarm only affects alarm")
    reset_world()
    detect("remind me to pay priya at 8pm $25", "chat_D")
    # Dismiss only alarm
    r = client.post("/popup-dismissed/chat_D?intent=alarm")
    body = r.json()
    check("dismissed response scoped to alarm",
          body["intent"] == "alarm" and body["chat_state"] == "dismissed")
    check("alarm state=dismissed",
          tracker("chat_D", "alarm")["state"] == "dismissed")
    check("money state still cooldown (not dismissed)",
          tracker("chat_D", "money")["state"] == "cooldown")

    # Invalid intent -> 400
    r = client.post("/popup-dismissed/chat_D?intent=bogus")
    check("invalid intent -> 400", r.status_code == 400)


def scenario_25_reset_cooldown_scoped():
    header(25, "/reset-cooldown with intent= clears just that intent")
    reset_world()
    detect("remind me to pay priya at 8pm $25", "chat_R")
    check("both entries present",
          tracker("chat_R", "money") is not None
          and tracker("chat_R", "alarm") is not None)

    # Scoped reset
    r = client.post("/reset-cooldown/chat_R?intent=money")
    body = r.json()
    check("reset response scoped to money",
          body["intent"] == "money" and body["existed"] is True)
    check("money entry removed",
          tracker("chat_R", "money") is None)
    check("alarm entry kept",
          tracker("chat_R", "alarm") is not None)

    # Unscoped reset clears the rest
    r = client.post("/reset-cooldown/chat_R")
    body = r.json()
    check("unscoped reset reports cleared alarm",
          "alarm" in body.get("cleared_intents", []))
    check("no entries left for chat_R",
          tracker("chat_R", "alarm") is None)


def scenario_26_intents_array_shape():
    header(26, "intents[] shape + payload/targeting fields on every intent")
    reset_world()
    r = detect("remind me to venmo priya $25 at 8pm", "chat_S")
    intents = r.get("intents", [])
    check("intents[] non-empty", len(intents) >= 2)

    for it in intents:
        check(f"{it['type']}: has confidence",
              isinstance(it.get("confidence"), (int, float)))
        check(f"{it['type']}: has should_popup bool",
              isinstance(it.get("should_popup"), bool))
        check(f"{it['type']}: has payload dict",
              isinstance(it.get("payload"), dict))
        check(f"{it['type']}: has targeting dict",
              isinstance(it.get("targeting"), dict))
        for tk in ("addressee", "third_party", "is_self", "is_mutual"):
            check(f"{it['type']}: targeting.{tk} present",
                  tk in it["targeting"])


def scenario_27_targeting_extraction():
    header(27, "Targeting signals: addressee / third_party / is_self / is_mutual")
    reset_world()
    # Self-reminder
    r = detect("remind me to call mom", "chat_T1")
    ai = intent_of(r, "alarm")
    check("self-reminder: is_self=True",
          ai and ai["targeting"]["is_self"] is True)
    check("self-reminder: third_party=mom",
          ai and ai["targeting"]["third_party"] == "mom")
    check("self-reminder: is_mutual=False",
          ai and ai["targeting"]["is_mutual"] is False)

    # Mutual / group event
    r = detect("team sync friday 4pm", "chat_T2")
    ci = intent_of(r, "calendar")
    check("team sync: is_mutual=True",
          ci and ci["targeting"]["is_mutual"] is True)
    check("team sync: is_self=False",
          ci and ci["targeting"]["is_self"] is False)


def scenario_28_maps_intent_fires():
    header(28, "Maps intent fires with place payload")
    reset_world()
    r = detect("heading to dolores park", "chat_M")
    mi = intent_of(r, "maps")
    check("maps intent detected", mi is not None)
    check("maps should_popup=True", mi and mi["should_popup"] is True)
    check("maps payload has place",
          mi and mi["payload"].get("place") is not None)
    # Any reasonable extraction that contains "dolores" passes
    place = (mi and mi["payload"].get("place") or "").lower()
    check("place contains 'dolores'", "dolores" in place,
          f"got place={place!r}")


def scenario_29_alarm_new_time_overrides_cooldown():
    header(29, "Distinct new time for alarm overrides its own cooldown")
    reset_world()
    detect("remind me at 10pm to take meds", "chat_AL")
    rewind_ts("chat_AL", 30, intent="alarm")
    # Different time -> should pop again
    r = detect("actually remind me at 11pm instead", "chat_AL")
    ai = intent_of(r, "alarm")
    check("alarm re-pops on new time",
          ai and ai["should_popup"] is True,
          f"got reason={ai and ai['suppressed_reason']}")
    check("alarm popup_count=2",
          tracker("chat_AL", "alarm")["popup_count"] == 2)


def scenario_30_contact_phone_extraction():
    header(30, "Contact intent extracts phone number")
    reset_world()
    r = detect("hey akash save this number +1 415-555-1234", "chat_C")
    ci = intent_of(r, "contact")
    check("contact intent detected", ci is not None)
    check("contact should_popup=True", ci and ci["should_popup"] is True)
    phone = ci and ci["payload"].get("phone")
    check("phone normalized to +1 415 555 1234",
          phone == "+1 415 555 1234",
          f"got {phone!r}")
    check("addressee=akash",
          ci and ci["targeting"]["addressee"] == "akash")


def scenario_31_health_reports_multi_intent():
    header(31, "/health reports num_labels=5 and all 5 intents")
    r = client.get("/health").json()
    check("num_labels=5", r["num_labels"] == 5)
    check("intents list is all 5",
          set(r["intents"]) == set(paychat_app.INTENTS))


def scenario_32_intent_specific_cooldown_suppression():
    header(32, "Same intent-fire twice in a row -> second is suppressed per intent")
    reset_world()
    detect("meeting at 3pm tomorrow", "chat_CAL")
    r = detect("meeting at 3pm tomorrow", "chat_CAL")  # same event, same payload
    ci = intent_of(r, "calendar")
    check("calendar 2nd fire suppressed",
          ci and ci["should_popup"] is False)
    check("calendar reason=cooldown_active",
          ci and ci["suppressed_reason"] == "cooldown_active")
    # Money wasn't involved — money back-compat flat fields should stay false
    check("back-compat is_money=False", r["is_money"] is False)


# ── Run all ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PayChat multi-intent popup cooldown -- end-to-end behavior tests")
    print("=" * 70)

    scenarios = [
        # Money back-compat (original 20)
        scenario_1_first_money_fires,
        scenario_2_cooldown_suppresses_follow_ups,
        scenario_3_non_money_doesnt_affect_cooldown,
        scenario_4_new_amount_overrides_cooldown,
        scenario_5_same_amount_stays_suppressed,
        scenario_6_cooldown_expires_naturally,
        scenario_7_payment_complete_clears_cooldown,
        scenario_8_after_grace_new_topic_pops,
        scenario_9_dismissed_has_longer_cooldown,
        scenario_10_dismissed_holds_past_normal_cooldown,
        scenario_11_chats_are_isolated,
        scenario_12_no_chat_id_lets_through,
        scenario_13_reset_cooldown_clears,
        scenario_14_chat_state_endpoint,
        scenario_15_metrics_track_accurately,
        scenario_16_eviction_of_stale_entries,
        scenario_17_websocket_same_contract,
        scenario_18_empty_text_rejected,
        scenario_19_payment_complete_on_unknown_chat,
        scenario_20_full_conversation_trace,
        # Multi-intent (v2)
        scenario_21_per_intent_cooldown_isolation,
        scenario_22_multi_intent_message_fires_both,
        scenario_23_payment_complete_scoped_to_money,
        scenario_24_dismiss_scoped_to_intent,
        scenario_25_reset_cooldown_scoped,
        scenario_26_intents_array_shape,
        scenario_27_targeting_extraction,
        scenario_28_maps_intent_fires,
        scenario_29_alarm_new_time_overrides_cooldown,
        scenario_30_contact_phone_extraction,
        scenario_31_health_reports_multi_intent,
        scenario_32_intent_specific_cooldown_suppression,
    ]
    for fn in scenarios:
        fn()

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 70)
    print(f"  {passed}/{len(results)} checks passed, {failed} failed")
    print("=" * 70)
    if failed:
        print("\nFailed checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}" + (f"  [{detail}]" if detail else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
