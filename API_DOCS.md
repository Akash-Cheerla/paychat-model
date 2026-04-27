# PayChat Intent Detection API — v2.0

Base URL: `http://<your-server>:8000`

PayChat detects **five actionable intents** in chat messages and tells the client app exactly when to surface a popup for each one:

| Intent | What it catches | Suggested action |
|--------|-----------------|------------------|
| `money` | "you owe me $20", "venmo me", "split the uber" | Open a payment app (Venmo, CashApp, UPI, etc.) |
| `alarm` | "remind me at 10pm", "wake me up at 6am tomorrow" | Set a system alarm |
| `contact` | "save this number +1 415-555-1234" | Save number to contacts |
| `calendar` | "meeting at 3pm tomorrow", "dinner sat 8pm" | Add calendar event |
| `maps` | "meet me at blue bottle", "heading to SFO" | Open location in a maps app |

All five run through the **same popup-cooldown policy, per chat, per intent** — so an alarm cooldown never blocks a money popup in the same conversation.

---

## `POST /detect`

Send a chat message, get back an array of detected intents and whether each should surface a popup right now.

### Request

```json
{
  "text": "remind me to venmo priya $25 at 8pm",
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
  "intents": [
    {
      "type": "money",
      "confidence": 0.9847,
      "should_popup": true,
      "suppressed_reason": null,
      "cooldown_remaining_seconds": 0,
      "chat_state": "cooldown",
      "payload": {
        "amount": "$25",
        "trigger_type": "payment_app",
        "direction": "request"
      },
      "targeting": {
        "addressee": null,
        "third_party": "priya",
        "is_self": true,
        "is_mutual": false
      }
    },
    {
      "type": "alarm",
      "confidence": 0.9912,
      "should_popup": true,
      "suppressed_reason": null,
      "cooldown_remaining_seconds": 0,
      "chat_state": "cooldown",
      "payload": {
        "label": "venmo priya",
        "time_iso": "2026-04-24T20:00",
        "time_phrase": "at 8pm"
      },
      "targeting": {
        "addressee": null,
        "third_party": "priya",
        "is_self": true,
        "is_mutual": false
      }
    }
  ],

  "is_money": true,
  "confidence": 0.9847,
  "trigger_type": "payment_app",
  "direction": "request",
  "detected_amount": "$25",
  "should_popup": true,
  "suppressed_reason": null,
  "cooldown_remaining_seconds": 0,
  "chat_state": "cooldown",

  "latency_ms": 28.4,
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

### Top-level response fields

| Field | Type | Description |
|-------|------|-------------|
| `intents` | array | **The v2 contract.** One entry per intent that cleared the confidence threshold. Empty array = nothing actionable detected. |
| `is_money`, `confidence`, `trigger_type`, `direction`, `detected_amount`, `should_popup`, `suppressed_reason`, `cooldown_remaining_seconds`, `chat_state` | mixed | **v1 back-compat.** Flat fields mirror the `money` intent if present (same values as `intents[type=money]`). New code should read `intents[]`; existing v1 code keeps working unchanged. |
| `latency_ms` | float | Model inference time in milliseconds |
| `chat_id`, `message_id`, `sender` | string or null | Passed through from the request |

### Per-intent fields (`intents[]` entries)

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | One of `money`, `alarm`, `contact`, `calendar`, `maps` |
| `confidence` | float | Sigmoid output for this head (0.0–1.0). Entries only appear above threshold (default 0.5). |
| `should_popup` | boolean | **The field your app reads.** `true` = fire the popup for this intent now. |
| `suppressed_reason` | string or null | Why the popup was suppressed. One of: `cooldown_active`, `recently_dismissed`, `post_payment_grace`. Null when `should_popup` is true. |
| `cooldown_remaining_seconds` | int | Seconds until a popup for this intent would be allowed again |
| `chat_state` | string | Tracker state for this (chat, intent). One of: `idle`, `cooldown`, `dismissed`, `post_payment`, `untracked` |
| `payload` | object | Data ready to drive a UI action for this intent. Shape varies per `type` — see below. |
| `targeting` | object | Who this message is addressed to. Shape below. |

### `payload` schemas per intent

#### `money`
```json
{ "amount": "$25", "trigger_type": "payment_app", "direction": "request" }
```
- `amount` — extracted dollar amount string, or null
- `trigger_type` — `owing_debt` | `bill_splitting` | `direct_amount` | `payment_app` | `general_money`
- `direction` — `request` | `offer` | `split` (determines who sees the popup)

#### `alarm`
```json
{ "label": "take meds", "time_iso": "2026-04-24T22:00", "time_phrase": "at 10pm", "seconds_from_now": 1200 }
```
- `label` — the task portion ("take meds"). May be null if not extractable.
- `time_iso` — ISO-8601 local time. Use to set a system alarm.
- `time_phrase` — raw phrase that was parsed
- `seconds_from_now` — only present for relative times ("in 20 min")

#### `contact`
```json
{ "phone": "+1 415 555 1234", "name_hint": "akash" }
```
- `phone` — normalized E.164-ish format with country code. US and India supported.
- `name_hint` — best-effort addressee pulled from the message. App should merge with chat context.

#### `calendar`
```json
{ "title": "meeting", "start_iso": "2026-04-25T15:00", "start_phrase": "at 3pm tomorrow", "duration_minutes": 30 }
```
- `title` — event title
- `start_iso` — ISO-8601 start time. Use to create a calendar event.
- `duration_minutes` — 30 for meetings/calls/standups, 60 for dinners/events

#### `maps`
```json
{ "place": "blue bottle on valencia" }
```
- `place` — free-text place string. Pass to a maps search.

### `targeting` schema (all intents)

```json
{ "addressee": "akash", "third_party": "priya", "is_self": true, "is_mutual": false }
```

- `addressee` — who the message is directed **at** within the chat (e.g. `"akash"` in "hey akash save this number"). Usually matches another chat member. Null if none detected.
- `third_party` — a person mentioned who isn't a chat participant (e.g. `"mom"` in "remind me to call mom"). Null if none detected.
- `is_self` — `true` if the sender is talking about themselves ("remind **me** to…"). Drives whose device should fire the popup for alarms/contacts/maps.
- `is_mutual` — `true` for group actions ("team sync friday 4pm", "everyone meet at the mission"). Drives "show popup to everyone in the chat" logic.

### The popup decision — `should_popup`

The model answers "is this about X?". The API answers "should we *actually* pop the sheet right now?". Those are not the same question.

In a chatty thread you might get 10 messages in 30 seconds — "you owe me 20", "bro pay up", "venmo me", "fr". The model says yes to all of them. If the app popped for every one, users would throw their phones. So the API holds **per-(chat, intent) state** and only returns `should_popup: true` when it's useful.

**The rules (apply independently per intent):**

- First intent-fire in a chat → popup fires. Tracker remembers `last_payload`.
- Follow-ups inside the cooldown window (5 min) → suppressed (`cooldown_active`).
- A **new distinct payload** for the same intent re-fires the popup. The distinctness key is:
  - `money` → `amount`
  - `alarm` → `time_iso` / `seconds_from_now`
  - `contact` → `phone`
  - `calendar` → `start_iso`
  - `maps` → `place`
- User dismisses → cooldown extends to 15 min (`recently_dismissed`).
- `/payment-complete` (money only) → cooldown clears, 60s grace window ignores "sent!" / "thanks" / "💸" messages (`post_payment_grace`).

**Per-intent isolation:** An alarm firing in chat X never affects the money cooldown in chat X. Each (chat_id, intent) pair gets its own state.

---

## App-side event reporting

Your app needs to tell the server when UI events happen so the cooldown state stays honest:

| Event | Endpoint | Scope |
|-------|----------|-------|
| Payment succeeded | `POST /payment-complete/{chat_id}` | money only |
| User dismissed a popup | `POST /popup-dismissed/{chat_id}?intent=<intent>` | specified intent (defaults to `money`) |
| Manual reset (admin/testing) | `POST /reset-cooldown/{chat_id}[?intent=<intent>]` | omit `intent` to clear all intents for the chat |

If you don't wire these up, the system still works — it falls back to timer-only behavior (5 min cooldown). Wiring them up is what makes it feel smart.

---

### Error responses

| Status | Body | When |
|--------|------|------|
| 400 | `{"detail": "text cannot be empty"}` | Empty or missing text |
| 400 | `{"detail": "unknown intent 'X'..."}` | Invalid `intent` query on `/popup-dismissed` or `/reset-cooldown` |
| 503 | `{"detail": "Model not loaded..."}` | Server still starting up (cold start ~20s) |

---

## `GET /health`

Health check. Use this for your load balancer or uptime monitoring.

```json
{
  "status":          "healthy",
  "device":          "cpu",
  "model_dir":       "/app/saved_model",
  "num_labels":      5,
  "intents":         ["money", "alarm", "contact", "calendar", "maps"],
  "loaded_at":       "2026-04-24T10:30:00",
  "version": {
    "trained_at":    "2026-04-24T...",
    "test_accuracy": 0.98,
    "test_f1":       0.97
  },
  "threshold":       0.5,
  "total_requests":  1284
}
```

`num_labels` will be `5` for the multi-intent model and `2` for a legacy money-only checkpoint. The server auto-detects and handles both.

---

## `GET /metrics`

Live stats. Good for dashboards.

```json
{
  "requests":              1284,
  "money_detected":        342,
  "intents_detected": {
    "money":    342,
    "alarm":    118,
    "contact":   47,
    "calendar": 203,
    "maps":     161
  },
  "detection_rate":        0.2664,
  "popups_fired":          418,
  "popups_suppressed":     453,
  "suppression_rate":      0.5201,
  "active_chat_trackers":  147,
  "avg_latency_ms":        28.5,
  "started_at":            "2026-04-24T10:30:00"
}
```

`suppression_rate` tells you what fraction of detections resulted in a suppressed popup. A healthy chatty-group app might run 0.4–0.7. Near-zero usually means your `chat_id` isn't being passed consistently — each message looks like a new chat, so nothing is ever a follow-up.

---

## `POST /reload`

Hot-reload the model without restarting the server. Call this after you swap model files on disk.

```json
{
  "status": "ok",
  "loaded_at": "2026-04-24T15:00:00",
  "num_labels": 5,
  "intents": ["money", "alarm", "contact", "calendar", "maps"],
  "version": { ... }
}
```

---

## `POST /payment-complete/{chat_id}`

**Money intent only.** Call this the moment a payment succeeds (Venmo webhook, in-app confirmation, UPI callback, etc.). Clears the money cooldown for this chat and opens a 60-second grace window so "sent!" / "thanks" / "💸" messages don't re-trigger the popup. Other intents' cooldowns in the same chat are untouched.

### Request

Body is optional — pass it if you want audit info in the logs.

```json
{ "amount": "$40", "payer": "rohit", "payee": "akash", "method": "venmo" }
```

### Response

```json
{
  "status": "ok",
  "chat_id": "room_abc123",
  "intent": "money",
  "chat_state": "post_payment",
  "previous_state": "cooldown",
  "grace_window_seconds": 60,
  "popup_count_total": 3
}
```

---

## `POST /popup-dismissed/{chat_id}?intent=<intent>`

Call this when the user dismisses a popup without completing the action (taps X, swipes away, closes the sheet). The cooldown extends to 15 minutes for that specific intent so we don't keep re-annoying them.

### Query params
| Param | Required | Default | Values |
|-------|----------|---------|--------|
| `intent` | No | `money` | one of `money`, `alarm`, `contact`, `calendar`, `maps` |

`intent` defaults to `money` so v1 callers (who always targeted the money popup) keep working without changes.

No body required.

### Response

```json
{
  "status": "ok",
  "chat_id": "room_abc123",
  "intent": "alarm",
  "chat_state": "dismissed",
  "previous_state": "cooldown",
  "cooldown_seconds": 900
}
```

---

## `POST /reset-cooldown/{chat_id}[?intent=<intent>]`

Force-clear the tracker. Admin / testing / power-user "snooze" only — normal flow should use `/payment-complete` or `/popup-dismissed`.

- `intent=<intent>` → clear that one intent's tracker
- Omit `intent` → clear **all** intents for this chat

### Response (scoped)

```json
{ "status": "ok", "chat_id": "room_abc123", "intent": "money", "existed": true, "chat_state": "idle" }
```

### Response (unscoped — all intents)

```json
{
  "status": "ok",
  "chat_id": "room_abc123",
  "cleared_intents": ["money", "alarm", "calendar"],
  "chat_state": "idle"
}
```

---

## `GET /chat-state/{chat_id}`

Inspect every intent's cooldown state for a chat. Useful during integration when you're wondering why a popup did or didn't fire.

### Response (tracker entries exist)

```json
{
  "chat_id": "room_abc123",
  "intents": {
    "money": {
      "state": "cooldown",
      "last_popup_at": "2026-04-24T10:45:10",
      "last_event_at": "2026-04-24T10:45:10",
      "last_payload": { "amount": "$40", "trigger_type": "owing_debt", "direction": "request" },
      "popup_count": 2,
      "suppression_count": 5,
      "cooldown_remaining_seconds": 187,
      "reason_for_current_state": "popup_just_fired"
    },
    "alarm": {
      "state": "dismissed",
      "last_popup_at": "2026-04-24T10:40:12",
      "last_event_at": "2026-04-24T10:41:50",
      "last_payload": { "label": "take meds", "time_iso": "2026-04-24T22:00" },
      "popup_count": 1,
      "suppression_count": 2,
      "cooldown_remaining_seconds": 778,
      "reason_for_current_state": "user_dismissed_popup"
    }
  }
}
```

### Response (no tracker entries yet)

```json
{
  "chat_id": "room_abc123",
  "state": "idle",
  "intents": {},
  "message": "no tracker entries — next message will popup for any detected intent"
}
```

---

## WebSocket: `ws://<server>:8000/ws/detect`

Real-time detection. Same logic as `POST /detect` but over a persistent connection — same intents array, same cooldown policy, same back-compat flat fields.

**Send:**
```json
{"text": "remind me to venmo priya $25 at 8pm", "chat_id": "room_abc", "sender": "akash"}
```

**Receive:**
```json
{
  "text": "remind me to venmo priya $25 at 8pm",
  "chat_id": "room_abc",
  "sender": "akash",
  "intents": [
    { "type": "money",  "should_popup": true, "payload": {...}, "targeting": {...} },
    { "type": "alarm",  "should_popup": true, "payload": {...}, "targeting": {...} }
  ],
  "venmo_detection": {
    "is_money": true,
    "confidence": 0.9912,
    "trigger_type": "payment_app",
    "direction": "request",
    "detected_amount": "$25",
    "should_popup": true,
    "suppressed_reason": null,
    "cooldown_remaining_seconds": 0,
    "chat_state": "cooldown",
    "latency_ms": 28.9
  }
}
```

The `venmo_detection` object is the v1-back-compat mirror of the money intent. New code should read `intents[]`.
