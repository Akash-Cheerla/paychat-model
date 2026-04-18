"""
End-to-end tests for popup cooldown behavior.

Runs without loading the real DistilBERT model -- we mock run_inference so the
cooldown logic itself is what gets exercised. Time is fast-forwarded by
mutating tracker timestamps directly, so the whole suite runs in <1 second.

Run:  python tests/test_cooldown.py
"""
import os
import re
import sys
import time

# Make sibling modules importable when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as paychat_app  # noqa: E402


# ── Mock inference ──────────────────────────────────────────────────────────
def mock_inference(text: str):
    """Simulate the model. Triggers on common money keywords. Extracts $N amounts."""
    t = text.lower()
    money_kw = ["$", " money", "venmo", "cashapp", "zelle", "owe", "pay me", "pay up",
                "split", "bucks", "dollar", "send me"]
    is_money = any(k in t for k in money_kw)

    m = re.search(r"\$\d+(?:\.\d{1,2})?", text)
    amount = m.group(0) if m else None

    trigger = direction = None
    if is_money:
        if any(w in t for w in ["venmo", "cashapp", "zelle"]):
            trigger = "payment_app"
        elif "owe" in t:
            trigger = "owing_debt"
        elif "split" in t:
            trigger = "bill_splitting"
        elif amount:
            trigger = "direct_amount"
        else:
            trigger = "general_money"

        if any(w in t for w in ["i owe", "i'll pay", "i'll send", "let me pay"]):
            direction = "offer"
        elif "split" in t:
            direction = "split"
        else:
            direction = "request"

    # Stats updates mirror the real run_inference so /metrics stays honest
    paychat_app.stats["requests"] += 1
    paychat_app.stats["_latency_sum"] += 1.0
    paychat_app.stats["avg_latency_ms"] = (
        paychat_app.stats["_latency_sum"] / paychat_app.stats["requests"]
    )
    if is_money:
        paychat_app.stats["money_detected"] += 1

    return {
        "is_money": is_money,
        "confidence": 0.99 if is_money else 0.01,
        "trigger_type": trigger,
        "direction": direction,
        "detected_amount": amount,
        "latency_ms": 1.0,
    }


# Bypass real model loading
paychat_app.load_model = lambda *a, **kw: None
paychat_app.model_state["model"] = "MOCK"
paychat_app.model_state["tokenizer"] = "MOCK"
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
        "requests": 0, "money_detected": 0,
        "popups_fired": 0, "popups_suppressed": 0,
        "avg_latency_ms": 0.0, "_latency_sum": 0.0,
    })


def detect(text, chat_id=None, sender="akash"):
    payload = {"text": text, "sender": sender}
    if chat_id:
        payload["chat_id"] = chat_id
    r = client.post("/detect", json=payload)
    assert r.status_code == 200, f"/detect failed: {r.status_code} {r.text}"
    return r.json()


def rewind_ts(chat_id, seconds):
    """Simulate `seconds` of wall-clock passing by backdating the tracker entry."""
    e = paychat_app.popup_tracker.get(chat_id)
    if not e:
        return
    e["last_popup_ts"] -= seconds
    e["last_event_ts"] -= seconds


# ════════════════════════════════════════════════════════════════════════════
#  SCENARIOS
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
    check("tracker entry created", "chat_A" in paychat_app.popup_tracker)
    check("popup_count=1",
          paychat_app.popup_tracker["chat_A"]["popup_count"] == 1)


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
        check(f"'{msg}' -> remaining > 0", r["cooldown_remaining_seconds"] > 0)
    check("popup_count still 1",
          paychat_app.popup_tracker["chat_A"]["popup_count"] == 1)
    check("suppression_count == 3",
          paychat_app.popup_tracker["chat_A"]["suppression_count"] == 3)


def scenario_3_non_money_doesnt_affect_cooldown():
    header(3, "Non-money chatter is ignored, doesn't touch cooldown")
    reset_world()
    detect("you owe me $40", "chat_A")
    before = dict(paychat_app.popup_tracker["chat_A"])
    r = detect("yeah yeah my bad lemme grab my phone", "chat_A")
    check("is_money=False on non-money", r["is_money"] is False)
    check("should_popup=False", r["should_popup"] is False)
    check("reason=not_money", r["suppressed_reason"] == "not_money")
    after = paychat_app.popup_tracker["chat_A"]
    check("tracker unchanged by non-money msg",
          before["last_popup_ts"] == after["last_popup_ts"]
          and before["popup_count"] == after["popup_count"])


def scenario_4_new_amount_overrides_cooldown():
    header(4, "Distinct new $ amount in same chat overrides cooldown")
    reset_world()
    detect("you owe me $40", "chat_A")
    # Simulate just 30s later -- well within cooldown
    rewind_ts("chat_A", 30)
    r = detect("oh also you owe me $15 for gas", "chat_A")
    check("should_popup=True on new amount", r["should_popup"] is True,
          f"got should_popup={r['should_popup']}, reason={r.get('suppressed_reason')}")
    check("detected_amount=$15", r["detected_amount"] == "$15")
    check("popup_count=2",
          paychat_app.popup_tracker["chat_A"]["popup_count"] == 2)
    check("last_amount=$15",
          paychat_app.popup_tracker["chat_A"]["last_amount"] == "$15")


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
    # Fast-forward past the 5-min window
    rewind_ts("chat_A", paychat_app.POPUP_COOLDOWN_SECONDS + 10)
    r = detect("pay up", "chat_A")
    check("should_popup=True after cooldown expires",
          r["should_popup"] is True,
          f"got reason={r.get('suppressed_reason')}")
    check("popup_count=2",
          paychat_app.popup_tracker["chat_A"]["popup_count"] == 2)


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

    # Victory-lap message during grace
    r = detect("sent! [money]", "chat_A")
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
    paychat_app.popup_tracker["chat_A"]["last_event_ts"] -= (
        paychat_app.POST_PAYMENT_GRACE_SECONDS + 5
    )
    r = detect("also you owe me $15 for gas", "chat_A")
    check("should_popup=True after grace", r["should_popup"] is True,
          f"got reason={r.get('suppressed_reason')}")
    check("popup_count=2",
          paychat_app.popup_tracker["chat_A"]["popup_count"] == 2)


def scenario_9_dismissed_has_longer_cooldown():
    header(9, "User dismissed -> 15-min cooldown (vs 5-min default)")
    reset_world()
    detect("you owe me $40", "chat_A")
    r = client.post("/popup-dismissed/chat_A")
    check("/popup-dismissed returns 200", r.status_code == 200)
    body = r.json()
    check("state=dismissed", body["chat_state"] == "dismissed")
    check("cooldown_seconds=900", body["cooldown_seconds"] == 900)

    # Suppressed with different reason
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
    # Fast-forward past the normal 5-min cooldown but not the 15-min dismissed one
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
          "chat_A" in paychat_app.popup_tracker
          and "chat_B" in paychat_app.popup_tracker)
    check("chat_A still in cooldown",
          paychat_app.popup_tracker["chat_A"]["state"] == "cooldown")


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
    check("entry exists", "chat_A" in paychat_app.popup_tracker)
    r = client.post("/reset-cooldown/chat_A")
    check("reset returns 200", r.status_code == 200)
    body = r.json()
    check("existed=True", body["existed"] is True)
    check("entry removed", "chat_A" not in paychat_app.popup_tracker)
    # Next money msg in that chat should pop
    r = detect("you owe me $40 bro", "chat_A")
    check("pops again after reset", r["should_popup"] is True)


def scenario_14_chat_state_endpoint():
    header(14, "GET /chat-state/{chat_id} reports accurate tracker state")
    reset_world()
    # Unknown chat
    r = client.get("/chat-state/unknown_chat")
    body = r.json()
    check("unknown chat -> state=idle", body["state"] == "idle")
    check("unknown chat has message field", "message" in body)

    # After a popup
    detect("you owe me $40", "chat_A")
    r = client.get("/chat-state/chat_A")
    body = r.json()
    check("state=cooldown", body["state"] == "cooldown")
    check("popup_count=1", body["popup_count"] == 1)
    check("last_amount=$40", body["last_amount"] == "$40")
    check("cooldown_remaining_seconds ~300",
          250 <= body["cooldown_remaining_seconds"] <= 300,
          f"got {body['cooldown_remaining_seconds']}")


def scenario_15_metrics_track_accurately():
    header(15, "/metrics counters reflect fire/suppression correctly")
    reset_world()
    detect("you owe me $40", "chat_A")         # fire
    detect("pay up bro", "chat_A")             # suppress
    detect("venmo me fr", "chat_A")            # suppress
    detect("what's for dinner?", "chat_A")     # not_money, no impact on popup counters
    detect("hey split ubers?", "chat_B")       # fire (different chat)

    r = client.get("/metrics").json()
    check("popups_fired=2", r["popups_fired"] == 2,
          f"got {r['popups_fired']}")
    check("popups_suppressed=2", r["popups_suppressed"] == 2,
          f"got {r['popups_suppressed']}")
    check("suppression_rate=0.5", abs(r["suppression_rate"] - 0.5) < 0.001,
          f"got {r['suppression_rate']}")
    check("active_chat_trackers=2", r["active_chat_trackers"] == 2)
    check("money_detected=4", r["money_detected"] == 4,
          f"got {r['money_detected']}")


def scenario_16_eviction_of_stale_entries():
    header(16, "Stale tracker entries evicted after TRACKER_EVICTION_SECONDS")
    reset_world()
    detect("you owe me $40", "old_chat")
    detect("you owe me $50", "new_chat")
    # Make old_chat stale
    paychat_app.popup_tracker["old_chat"]["last_event_ts"] -= (
        paychat_app.TRACKER_EVICTION_SECONDS + 60
    )
    # Any detect call triggers opportunistic cleanup
    detect("random non-money", "new_chat")
    check("stale entry evicted",
          "old_chat" not in paychat_app.popup_tracker)
    check("fresh entry kept",
          "new_chat" in paychat_app.popup_tracker)


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

        # Follow-up over the same ws
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
    r = detect("sent! [money]", "trip_chat")
    check("#7 'sent!' not_money", r["suppressed_reason"] == "not_money")
    r = detect("got it thanks bro", "trip_chat")
    check("#8 'thanks' not_money", r["suppressed_reason"] == "not_money")

    # 9. Simulate 45 min passing -> new topic in same chat
    paychat_app.popup_tracker["trip_chat"]["last_event_ts"] -= 2700
    paychat_app.popup_tracker["trip_chat"]["last_popup_ts"] -= 2700
    r = detect("oh also u owe me $15 for gas", "trip_chat")
    check("#9 new amount after 45min pops", r["should_popup"] is True,
          f"reason={r.get('suppressed_reason')}")

    # 10. User dismisses, then more nags are suppressed harder
    client.post("/popup-dismissed/trip_chat")
    r = detect("bro the gas money", "trip_chat")
    check("#10 post-dismiss suppressed",
          r["suppressed_reason"] == "recently_dismissed")

    # Final counters
    m = client.get("/metrics").json()
    check("2 popups fired across conversation", m["popups_fired"] == 2,
          f"got {m['popups_fired']}")


# ── Run all ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PayChat popup cooldown -- end-to-end behavior tests")
    print("=" * 70)

    for fn in [
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
    ]:
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
