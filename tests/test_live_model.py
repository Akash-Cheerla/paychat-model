"""
Live end-to-end smoke test — loads the REAL saved_model/ and runs /detect
through FastAPI's TestClient. Unlike test_cooldown.py (which mocks inference),
this exercises the actual model + tokenizer + extractors together.

Run:  python tests/test_live_model.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as paychat_app  # triggers model load via FastAPI lifespan

from fastapi.testclient import TestClient


PASS = "[PASS]"
FAIL = "[FAIL]"
_results = []


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  {status} {name}" + (f"   -- {detail}" if detail and not cond else ""))
    _results.append((name, bool(cond), detail))


def main():
    print("=" * 70)
    print("  PayChat LIVE model smoke test")
    print("=" * 70)
    print(f"\nLoading model from {paychat_app.MODEL_DIR}...")
    t0 = time.time()

    # TestClient triggers the FastAPI lifespan which loads the model
    with TestClient(paychat_app.app) as client:
        load_ms = (time.time() - t0) * 1000
        print(f"Model loaded in {load_ms:.0f}ms\n")

        # --Health --
        r = client.get("/health").json()
        print(f"/health: status={r['status']} num_labels={r['num_labels']} intents={r['intents']}\n")
        check("health status=healthy", r["status"] == "healthy")
        check("num_labels=5", r["num_labels"] == 5)
        check("intents match", r["intents"] == ["money", "alarm", "contact", "calendar", "maps"])

        # --Per-intent positive cases --
        cases = [
            ("you owe me $40 from the airbnb",              "money"),
            ("venmo me $25 for pizza",                      "money"),
            ("split the uber 30 bucks each",                "money"),
            ("remind me to take meds at 10pm",              "alarm"),
            ("wake me up at 6am tomorrow",                  "alarm"),
            ("hey akash save this number +1 415-555-1234",  "contact"),
            ("here's my number 9876543210",                 "contact"),
            ("meeting at 3pm tomorrow",                     "calendar"),
            ("team sync friday 4pm",                        "calendar"),
            ("heading to dolores park",                     "maps"),
            ("meet me at blue bottle on valencia",          "maps"),
        ]

        print("--Per-intent positive detections --")
        for text, expected in cases:
            r = client.post("/detect", json={"text": text, "chat_id": f"live_{expected}"}).json()
            types = {i["type"] for i in r["intents"]}
            check(f"{text[:45]:<45s} -> {expected}",
                  expected in types,
                  f"got intents={types}")

        # --Multi-intent --
        print("\n--Multi-intent --")
        r = client.post("/detect",
                        json={"text": "remind me to venmo priya $25 at 8pm",
                              "chat_id": "live_multi"}).json()
        types = {i["type"] for i in r["intents"]}
        check("multi-intent: money fired", "money" in types)
        check("multi-intent: alarm fired", "alarm" in types)
        money_payload = next((i["payload"] for i in r["intents"] if i["type"] == "money"), {})
        alarm_payload = next((i["payload"] for i in r["intents"] if i["type"] == "alarm"), {})
        check("money.amount=$25", money_payload.get("amount") == "$25",
              f"got {money_payload.get('amount')}")
        check("alarm.time_iso present", alarm_payload.get("time_iso") is not None,
              f"got {alarm_payload.get('time_iso')}")

        # --Negatives (should NOT fire any intent) --
        print("\n--Negatives --")
        negatives = [
            "whats up",
            "lol yeah",
            "got it thanks",
            "omw",
        ]
        for text in negatives:
            r = client.post("/detect", json={"text": text, "chat_id": "neg"}).json()
            types = {i["type"] for i in r["intents"]}
            check(f"{text!r:30s} -> no intent", len(types) == 0,
                  f"got intents={types}")

        # --Latency --
        print("\n--Latency (single-request inference) --")
        t0 = time.time()
        r = client.post("/detect", json={"text": "you owe me $40", "chat_id": "lat"}).json()
        elapsed = (time.time() - t0) * 1000
        print(f"  /detect wall-clock: {elapsed:.1f}ms  (model latency_ms: {r['latency_ms']:.1f})")
        check("latency under 500ms (CPU)", elapsed < 500)

        # --Cooldown smoke --
        print("\n--Cooldown (real model) --")
        client.post("/reset-cooldown/cd_test")
        r1 = client.post("/detect", json={"text": "you owe me $40", "chat_id": "cd_test"}).json()
        r2 = client.post("/detect", json={"text": "bro pay up", "chat_id": "cd_test"}).json()
        check("1st money msg pops",  r1["should_popup"] is True)
        check("2nd money msg suppressed", r2["should_popup"] is False)
        check("2nd msg reason=cooldown_active",
              r2["suppressed_reason"] == "cooldown_active",
              f"got {r2['suppressed_reason']}")

        # --/metrics shape --
        print("\n--/metrics --")
        m = client.get("/metrics").json()
        check("metrics has intents_detected dict",
              isinstance(m.get("intents_detected"), dict)
              and set(m["intents_detected"]) == {"money", "alarm", "contact", "calendar", "maps"})
        print(f"  requests={m['requests']} popups_fired={m['popups_fired']} "
              f"popups_suppressed={m['popups_suppressed']} avg_latency={m['avg_latency_ms']}ms")
        print(f"  per-intent detections: {m['intents_detected']}")

    # --Summary --
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print("\n" + "=" * 70)
    print(f"  {passed}/{len(_results)} live-model checks passed, {failed} failed")
    print("=" * 70)
    if failed:
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}" + (f"  [{detail}]" if detail else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
