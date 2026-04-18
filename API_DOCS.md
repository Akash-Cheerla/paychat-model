# PayChat Money Detection API

Base URL: `http://<your-server>:8000`

---

## `POST /detect`

Send a chat message, get back whether it's about money and what to do with it.

### Request

```json
{
  "text": "you owe me $20 from last night",
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | The chat message to analyze |
| `chat_id` | string | No | Your conversation/room ID — passed through unchanged |
| `message_id` | string | No | Your message ID — passed through unchanged |
| `sender` | string | No | Who sent the message — passed through unchanged |

### Response

```json
{
  "is_money": true,
  "confidence": 0.9847,
  "trigger_type": "owing_debt",
  "direction": "request",
  "detected_amount": "$20",
  "should_popup": true,
  "suppressed_reason": null,
  "cooldown_remaining_seconds": 0,
  "chat_state": "cooldown",
  "latency_ms": 245.3,
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `is_money` | boolean | `true` if the message is about money/payments |
| `confidence` | float | Model confidence (0.0 to 1.0). Threshold is 0.65 |
| `trigger_type` | string or null | Why it triggered. One of: `owing_debt`, `bill_splitting`, `direct_amount`, `payment_app`, `general_money`. Null if `is_money` is false |
| `direction` | string or null | Who should see the popup. One of: `request`, `offer`, `split`. Null if `is_money` is false |
| `detected_amount` | string or null | Extracted dollar amount (e.g. `"$20"`, `"$15.50"`). Null if no amount found |
| `should_popup` | boolean | **The one field the app should actually read.** `true` = show the Venmo popup now. `false` = stay quiet. |
| `suppressed_reason` | string or null | Why the popup was suppressed. One of: `not_money`, `cooldown_active`, `recently_dismissed`, `post_payment_grace`. Null when `should_popup` is true. |
| `cooldown_remaining_seconds` | int | Seconds until a popup would be allowed again. `0` when `should_popup` is true. |
| `chat_state` | string | Current tracker state for this chat. One of: `idle`, `cooldown`, `dismissed`, `post_payment`, `untracked`. |
| `latency_ms` | float | How long inference took in milliseconds |
| `chat_id` | string or null | Passed through from request |
| `message_id` | string or null | Passed through from request |
| `sender` | string or null | Passed through from request |

### The popup decision — `should_popup`

The model answers "is this about money?". The API answers "should we *actually* pop the Venmo sheet right now?". Those are not the same question.

In a money conversation you might get 10 messages in 30 seconds — "you owe me 20", "bro pay up", "venmo me", "fr". The model says yes to all of them. If the app popped for every one, users would throw their phones. So the API holds a per-chat state machine and only returns `should_popup: true` when it's actually useful.

**The rules:**

- First money message for a chat → popup fires.
- Follow-ups inside the cooldown window (5 min) → suppressed (`cooldown_active`).
- If a new, distinct `$` amount shows up in the same chat → popup fires again (it's a new transaction).
- If the user dismisses the popup → cooldown extends to 15 min (`recently_dismissed`).
- If payment completes → cooldown clears, brief 60s grace window to ignore "sent!" / "thanks" messages (`post_payment_grace`). After grace, chat is fully idle again.

**Your integration is just one line.** Read `should_popup`. If true, pop. If false, don't. Everything else in the response is diagnostic — useful for logs, dashboards, debugging — but not required.

### Popup lifecycle — app-side events to report

Your app needs to tell us when certain UI events happen so we can update the state machine. Three endpoints:

| Event | Endpoint | When to call |
|-------|----------|--------------|
| Payment succeeded | `POST /payment-complete/{chat_id}` | Venmo/CashApp webhook confirmed, or in-app payment flow reached success screen |
| User dismissed popup | `POST /popup-dismissed/{chat_id}` | User tapped X / closed the popup without paying |
| Manual reset (admin/testing) | `POST /reset-cooldown/{chat_id}` | You want to force-clear the cooldown for any reason |

If you don't wire these up, the system still works — it just falls back to timer-only behavior (5 min cooldown always). Wiring them up is what makes it feel smart.

### Direction field — who gets the popup

| Direction | Meaning | Show popup to |
|-----------|---------|--------------|
| `request` | Sender is asking for money ("you owe me $20") | Recipients (everyone except sender) |
| `offer` | Sender is offering to pay ("I'll venmo you") | Sender |
| `split` | Mutual split ("let's split the bill") | Everyone in the chat |

### Trigger types

| Trigger Type | Example Messages |
|-------------|-----------------|
| `owing_debt` | "you owe me $20", "pay me back", "where's my money" |
| `bill_splitting` | "let's split the bill", "your share is $15", "go halves?" |
| `direct_amount` | "send me $50", "that'll be $30", "I need $20 from you" |
| `payment_app` | "venmo me", "cashapp me $20", "zelle me" |
| `general_money` | "my treat", "I got you", "drinks on me", "spot me" |

### Error responses

| Status | Body | When |
|--------|------|------|
| 400 | `{"detail": "text cannot be empty"}` | Empty or missing text |
| 503 | `{"detail": "Model not loaded..."}` | Server still starting up (cold start ~20s) |

---

## `GET /health`

Health check. Use this for your load balancer or uptime monitoring.

```json
{
  "status": "healthy",
  "device": "cpu",
  "model_dir": "/app/saved_model",
  "loaded_at": "2026-04-11T10:30:00",
  "version": {
    "trained_at": "2026-04-12T...",
    "test_accuracy": 1.0,
    "test_f1": 1.0
  },
  "threshold": 0.65,
  "total_requests": 1284
}
```

---

## `GET /metrics`

Live stats. Good for dashboards.

```json
{
  "requests": 1284,
  "money_detected": 342,
  "detection_rate": 0.2664,
  "popups_fired": 118,
  "popups_suppressed": 224,
  "suppression_rate": 0.6550,
  "active_chat_trackers": 47,
  "avg_latency_ms": 231.5,
  "started_at": "2026-04-11T10:30:00"
}
```

`suppression_rate` tells you what fraction of detected money messages resulted in a suppressed popup. A healthy chatty-group chat might run 0.6–0.8 — that means the cooldown is earning its keep. Near-zero means either every chat has a single money message (fine) or your `chat_id` isn't being passed consistently (check that first).

---

## `POST /reload`

Hot-reload the model without restarting the server. Call this after you update the model files on disk.

```json
{
  "status": "ok",
  "loaded_at": "2026-04-11T15:00:00",
  "version": { ... }
}
```

---

## `POST /payment-complete/{chat_id}`

Call this the moment a payment succeeds (Venmo webhook, in-app confirmation, whatever your source is). It clears the chat's cooldown so future money topics can pop again, and briefly suppresses popups for 60 seconds so "sent!", "thanks", "💸" type victory-lap messages don't accidentally re-trigger.

### Request

Body is optional — pass it if you want audit info logged.

```json
{
  "amount": "$40",
  "payer": "rohit",
  "payee": "akash",
  "method": "venmo"
}
```

### Response

```json
{
  "status": "ok",
  "chat_id": "room_abc123",
  "chat_state": "post_payment",
  "previous_state": "cooldown",
  "grace_window_seconds": 60,
  "popup_count_total": 3
}
```

---

## `POST /popup-dismissed/{chat_id}`

Call this when the user actively dismisses the popup without paying (taps X, closes the sheet, etc.). We extend the cooldown to 15 minutes so we don't keep re-annoying them every time "money" appears in the chat.

No body required.

### Response

```json
{
  "status": "ok",
  "chat_id": "room_abc123",
  "chat_state": "dismissed",
  "previous_state": "cooldown",
  "cooldown_seconds": 900
}
```

---

## `POST /reset-cooldown/{chat_id}`

Force-clear the tracker for a chat. Admin / testing only — normal flow should use `/payment-complete` or `/popup-dismissed`.

### Response

```json
{
  "status": "ok",
  "chat_id": "room_abc123",
  "existed": true,
  "chat_state": "idle"
}
```

---

## `GET /chat-state/{chat_id}`

Inspect the cooldown tracker for a chat. Useful during integration when you're wondering why a popup did or didn't fire.

### Response (tracker entry exists)

```json
{
  "chat_id": "room_abc123",
  "state": "cooldown",
  "last_popup_at": "2026-04-18T10:45:10",
  "last_event_at": "2026-04-18T10:45:10",
  "last_amount": "$40",
  "last_trigger": "owing_debt",
  "popup_count": 2,
  "suppression_count": 5,
  "cooldown_remaining_seconds": 187,
  "reason_for_current_state": "popup_just_fired"
}
```

### Response (no tracker entry)

```json
{
  "chat_id": "room_abc123",
  "state": "idle",
  "message": "no tracker entry — next money message will popup"
}
```

---

## WebSocket: `ws://<server>:8000/ws/detect`

For real-time detection. Same logic as POST but over a persistent connection.

**Send:**
```json
{"text": "venmo me $20", "chat_id": "room_abc", "sender": "akash"}
```

**Receive:**
```json
{
  "text": "venmo me $20",
  "chat_id": "room_abc",
  "sender": "akash",
  "venmo_detection": {
    "is_money": true,
    "confidence": 0.9912,
    "trigger_type": "payment_app",
    "direction": "request",
    "detected_amount": "$20",
    "should_popup": true,
    "suppressed_reason": null,
    "cooldown_remaining_seconds": 0,
    "chat_state": "cooldown",
    "latency_ms": 189.4
  }
}
```

The WebSocket applies the exact same cooldown policy as `POST /detect` — same fields, same rules. The app team has one contract regardless of transport.

---

## Server Recommendations

| Scale | Instance | Specs | Cost | Latency |
|-------|----------|-------|------|---------|
| MVP (<500 users) | AWS `t3.medium` / GCP `e2-medium` | 2 vCPU, 4GB RAM | ~$30/mo | ~200-300ms |
| Growing (500-2000) | AWS `t3.large` / GCP `e2-standard-2` | 2 vCPU, 8GB RAM | ~$60/mo | ~150-250ms |
| Scale (2000+) | GPU instance or multiple CPU behind LB | varies | varies | ~20-50ms (GPU) |

CPU is fine for MVP. No GPU needed — model is only 67M parameters.

### Docker deployment

```bash
docker build -t paychat-model .
docker run -d -p 8000:8000 --name paychat paychat-model
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./saved_model` | Path to model files |
| `CONFIDENCE_THRESHOLD` | `0.65` | Min confidence to trigger (0.0-1.0) |
| `POPUP_COOLDOWN_SECONDS` | `300` | Quiet window after a popup fires (default 5 min) |
| `DISMISSED_COOLDOWN_SECONDS` | `900` | Longer quiet window if user dismissed the popup (default 15 min) |
| `POST_PAYMENT_GRACE_SECONDS` | `60` | Brief suppression right after `/payment-complete` so confirmation messages don't re-pop |
| `TRACKER_EVICTION_SECONDS` | `1800` | Drop in-memory chat tracker entries after this much idle time (default 30 min) |

### A note on scale

The cooldown tracker lives in memory on a single server. For MVP (single instance) this is fine and keeps things fast — no Redis dependency, no network hop. If you scale to multiple API instances behind a load balancer, the tracker needs to move to Redis so all instances share the same view of each chat's state. Swap the in-memory `popup_tracker` dict in `app.py` for a Redis client with the same keys and you're done.

---

## Quick test

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "you owe me $20", "chat_id": "test123"}'
```
