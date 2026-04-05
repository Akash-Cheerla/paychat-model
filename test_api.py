"""
Quick verification script for the PayChat Money Detection API.

Usage:
  1. Start the server:  uvicorn app:app --host 0.0.0.0 --port 8000
  2. Run this script:   python test_api.py
  3. All tests should pass with green checkmarks.

If running against a remote server:
  python test_api.py https://your-server.com
"""

import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
passed = 0
failed = 0


def test(name, method, path, body=None, check=None):
    global passed, failed
    url = f"{BASE_URL}{path}"
    try:
        if method == "POST":
            data = json.dumps(body).encode() if body else None
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        else:
            req = urllib.request.Request(url)

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        if check and not check(result):
            print(f"  FAIL  {name}")
            print(f"         Response: {json.dumps(result, indent=2)}")
            failed += 1
        else:
            print(f"  PASS  {name}")
            passed += 1
        return result

    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"         Error: {e}")
        failed += 1
        return None


print(f"\nPayChat API Test Suite")
print(f"Target: {BASE_URL}")
print(f"{'=' * 50}\n")

# Health check
print("[Health]")
test("Server is healthy",
     "GET", "/health",
     check=lambda r: r.get("status") in ("healthy", "ok"))

test("Model is loaded",
     "GET", "/health",
     check=lambda r: r.get("version") is not None and r["version"].get("test_accuracy") is not None)

# Detection — money messages
print("\n[Detection - Money Messages]")
test("'you owe me $25' -> is_money=true, direction=request",
     "POST", "/detect",
     body={"text": "you owe me $25", "chat_id": "test"},
     check=lambda r: r["is_money"] and r["direction"] == "request" and r["detected_amount"] == "$25")

test("'I'll venmo you $20' -> is_money=true, direction=offer",
     "POST", "/detect",
     body={"text": "I'll venmo you $20", "chat_id": "test"},
     check=lambda r: r["is_money"] and r["direction"] == "offer" and r["trigger_type"] == "payment_app")

test("'let's split dinner 3 ways' -> is_money=true, direction=split",
     "POST", "/detect",
     body={"text": "let's split dinner 3 ways", "chat_id": "test"},
     check=lambda r: r["is_money"] and r["direction"] == "split" and r["trigger_type"] == "bill_splitting")

test("'pay me back the 50$' -> is_money=true, amount=$50",
     "POST", "/detect",
     body={"text": "pay me back the 50$", "chat_id": "test"},
     check=lambda r: r["is_money"] and r["detected_amount"] == "$50")

test("'cashapp me for the tickets' -> is_money=true, payment_app",
     "POST", "/detect",
     body={"text": "cashapp me for the tickets", "chat_id": "test"},
     check=lambda r: r["is_money"] and r["trigger_type"] == "payment_app")

test("'do I owe you anything?' -> is_money=true, direction=offer",
     "POST", "/detect",
     body={"text": "do I owe you anything?", "chat_id": "test"},
     check=lambda r: r["is_money"] and r["direction"] == "offer")

test("'shall I send the remaining $30?' -> is_money=true, direction=offer",
     "POST", "/detect",
     body={"text": "shall I send the remaining $30?", "chat_id": "test"},
     check=lambda r: r["is_money"] and r["direction"] == "offer" and r["detected_amount"] == "$30")

# Detection — not money
print("\n[Detection - Non-Money Messages]")
test("'what time is dinner tonight' -> is_money=false",
     "POST", "/detect",
     body={"text": "what time is dinner tonight", "chat_id": "test"},
     check=lambda r: not r["is_money"])

test("'haha that's so funny' -> is_money=false",
     "POST", "/detect",
     body={"text": "haha that's so funny", "chat_id": "test"},
     check=lambda r: not r["is_money"])

test("'can you send me that link' -> is_money=false",
     "POST", "/detect",
     body={"text": "can you send me that link", "chat_id": "test"},
     check=lambda r: not r["is_money"])

test("'the stock market is up today' -> is_money=false",
     "POST", "/detect",
     body={"text": "the stock market is up today", "chat_id": "test"},
     check=lambda r: not r["is_money"])

# Passthrough fields
print("\n[Passthrough Fields]")
test("chat_id, message_id, sender echoed back",
     "POST", "/detect",
     body={"text": "you owe me $10", "chat_id": "room_abc", "message_id": "msg_123", "sender": "akash"},
     check=lambda r: r["chat_id"] == "room_abc" and r["message_id"] == "msg_123" and r["sender"] == "akash")

# Error handling
print("\n[Error Handling]")
try:
    req = urllib.request.Request(
        f"{BASE_URL}/detect",
        data=json.dumps({"text": ""}).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=10)
    print(f"  FAIL  Empty text returns 400 (got 200 instead)")
    failed += 1
except urllib.error.HTTPError as e:
    if e.code == 400:
        print(f"  PASS  Empty text returns 400")
        passed += 1
    else:
        print(f"  FAIL  Empty text returns 400 (got {e.code})")
        failed += 1

# Metrics
print("\n[Metrics]")
test("Metrics endpoint works",
     "GET", "/metrics",
     check=lambda r: "requests" in r and "money_detected" in r and r["requests"] > 0)

# Summary
print(f"\n{'=' * 50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed == 0:
    print("All tests passed!")
else:
    print(f"{failed} test(s) failed. Review output above.")
print()
