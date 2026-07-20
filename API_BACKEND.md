# PayChat v25 — Backend Integration

**For:** Samyak (Phoenix/Elixir backend)  
**Model:** DualHeadRoberta v25 (9 intents + response head, context-aware)  
**Repo:** https://github.com/Akash-Cheerla/paychat-model

---

## What's New (v25)

- **Response head** — ML classifier that understands responses to pending requests (ack, reject, future promise, question, already done, neutral). Replaces old regex-based classification.
- **Conversation state machine** — intents like money and ride fire on the *response*, not the request. Alice says "venmo me $20" → stored as pending. Bob says "sure" → money intent fires on Bob's message.
- **`triggered_by` field** — when an intent fires from a response, you now get the original requester's info (sender, text, message_id, slots). Critical for group chats.
- **Greeting guard** — bare greetings (hi, hey, yo, lol, etc.) are forced neutral before ML runs. Prevents false fires.

---

## Quick Start

```bash
git clone https://github.com/Akash-Cheerla/paychat-model.git
cd paychat-model
docker build -t paychat .
docker run -p 8000:8000 paychat
```

Without Docker:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
MODEL_DIR=./saved_model python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check: `GET /health` returns `{"status": "ok", ...}` once model is loaded (~15-20s on CPU).

---

## Endpoint: `POST /classify`

This is the only endpoint you need. Call it for every DM message.

### Request

```json
{
  "text": "venmo me 30 bucks for dinner",
  "room_id": "dm_12_45",
  "sender": "12",
  "context": [
    {"text": "hey are you free tonight?", "sender": "45"},
    {"text": "yeah lets grab food", "sender": "12"}
  ],
  "message_id": "msg_789"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | **Yes** | The message to classify |
| `room_id` | string | **Yes** | DM room ID — `dm_<lo>_<hi>` format. Needed for conversation state tracking. |
| `sender` | string | **Yes** | User ID of whoever sent this message. The state machine uses this to know who's responding to whose request. |
| `context` | object[] | No | Previous messages. Each object: `{"text": "...", "sender": "..."}`. Pass last 2-3 messages. If omitted, server tracks internally per room_id. |
| `message_id` | string | No | Message ID (echoed back, also stored with pending requests for `triggered_by`) |

**`sender` is required now.** Without it the server falls back to immediate-fire mode (no request→response tracking), which defeats the whole point of the state machine.

**`context` format changed.** Old format was a plain string array — that still works for backward compat but you lose sender info on context messages. New format is an array of `{"text": "...", "sender": "..."}` objects. Sender on context messages matters because the state machine needs to know who said what.

### Response

```json
{
  "intents": ["money"],
  "scores": {
    "money": 0.851,
    "ride": 0.021,
    "food_order": 0.038,
    "contact": 0.036,
    "alarm": 0.021,
    "reminder": 0.020,
    "calendar": 0.032,
    "bills": 0.039,
    "travel": 0.037
  },
  "slots": {
    "amount": "30 bucks",
    "recipient": null,
    "note": "dinner"
  },
  "money": {
    "detected_amount": "30 bucks",
    "trigger_type": "payment_app",
    "direction": "request"
  },
  "target": {
    "show_to": "others",
    "reason": "sender_requesting_payment"
  },
  "conversation_state": null,
  "lifecycle": null,
  "guardrails": null,
  "context_boosted": null,
  "latency_ms": 435.5,
  "chat_id": null,
  "message_id": "msg_789",
  "sender": "12"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `intents` | string[] | Fired intents (empty if nothing detected or if request is stored as pending) |
| `scores` | object | Confidence per intent (0.0–1.0) |
| `slots` | object \| null | Extracted entities (flat key-value) |
| `money` | object \| null | Money enrichment if money intent fired |
| `target` | object \| null | Who should see the popup |
| `conversation_state` | object \| null | State machine result — see below |
| `lifecycle` | object \| null | Cancel/defer/confirm state changes |
| `guardrails` | object \| null | Compliance flags (PCI, AML, phishing) |
| `latency_ms` | float | Inference time |

---

## Conversation State Machine (the big change)

For money and ride intents, the model doesn't fire immediately anymore. It works in two steps:

1. **Request** — "venmo me $20" → intent detected but stored as **pending**, not fired. `intents` comes back empty.
2. **Response** — "sure, sending now" → state machine detects this as an ack to the pending request → money intent **fires** on this message.

This prevents false positives. Someone saying "venmo me $20" isn't an action yet — it's a request. The action happens when someone responds.

### `conversation_state` field

Present on every response. Tells you what the state machine decided.

**When a request is stored as pending:**
```json
{
  "conversation_state": {
    "status": "pending",
    "pending_intents": ["money"],
    "response_type": "neutral",
    "reason": "new money request stored as pending"
  }
}
```
`intents` will be `[]` — nothing fires yet. Don't show a popup.

**When a response fires the intent:**
```json
{
  "conversation_state": {
    "status": "fired",
    "response_type": "positive_ack",
    "reason": "ML classified as ack to pending money",
    "triggered_by": {
      "sender": "12",
      "text": "venmo me 30 bucks for dinner",
      "message_id": "msg_789",
      "slots": {"amount": "30 bucks", "note": "dinner"}
    }
  }
}
```
`intents` will be `["money"]`. **Show the popup to the current sender** (the person who acked). `triggered_by.sender` tells you who originally requested — that's the payment recipient.

**When someone sets a reminder (future promise):**
```json
{
  "conversation_state": {
    "status": "reminder",
    "original_intent": "money",
    "response_type": "future_promise",
    "reason": "ML classified as future promise",
    "triggered_by": {
      "sender": "12",
      "text": "venmo me 30 bucks",
      "message_id": "msg_789",
      "slots": {"amount": "30 bucks"}
    }
  }
}
```
`intents` will be `["reminder"]`. Set a reminder for the responder to pay `triggered_by.sender`.

**When someone rejects:**
```json
{
  "conversation_state": {
    "status": "cancelled",
    "response_type": "rejection",
    "reason": "ML classified as rejection",
    "triggered_by": {
      "sender": "12",
      "text": "venmo me 30 bucks",
      "message_id": "msg_789"
    }
  }
}
```
`intents` will be `[]`. Request is cancelled. No popup.

**When a message is just neutral (no pending or unrelated):**
```json
{
  "conversation_state": {
    "status": "no_fire",
    "response_type": "neutral",
    "reason": "no pending requests in room"
  }
}
```

### `conversation_state.status` values

| Status | Meaning | `intents` | Action |
|--------|---------|-----------|--------|
| `"pending"` | Request stored, waiting for response | `[]` | Nothing — wait for response |
| `"fired"` | Response acknowledged a pending request | `["money"]` or `["ride"]` | Show popup to current sender |
| `"reminder"` | Response was a future promise | `["reminder"]` | Set reminder for current sender |
| `"cancelled"` | Response was a rejection | `[]` | Clear pending state |
| `"no_fire"` | Nothing relevant | `[]` | Nothing |

### `triggered_by` — who originally requested

Only present when status is `fired`, `reminder`, or `cancelled`. Tells you who made the original request that this message is responding to.

| Field | Type | Description |
|-------|------|-------------|
| `sender` | string | User ID of original requester |
| `text` | string | Original request text |
| `message_id` | string \| null | Original message ID (if you passed it) |
| `slots` | object \| null | Slots extracted from original request (amount, note, etc.) |

**This is how you know who to pay in a group chat.** In a 1:1 DM it's obvious (the other person). In a group chat, `triggered_by.sender` is the person who asked for money.

---

## `target.show_to` — Who Gets the Popup

| Value | Meaning | Example |
|-------|---------|---------|
| `"sender"` | Show popup to the person who sent the message | "book me an uber", "order pizza" |
| `"others"` | Show popup to everyone except sender | "venmo me 30", "you owe me" |
| `"group"` | Show to everyone | "let's split dinner" |

Note: when a pending request fires from a response, `target` is computed for the response message. So if Bob says "sure" and money fires, the target logic evaluates on "sure" which defaults to `show_to: "sender"` → show to Bob. That's correct — Bob is the one who needs to open Venmo.

---

## Full Flow Example

Here's a complete DM conversation and what each `/classify` call returns:

```
Alice (ID: 12): "venmo me 20 for lunch"
→ intents: [], conversation_state.status: "pending"
→ No popup. Request stored.

Bob (ID: 45): "Hi"
→ intents: [], conversation_state.status: "no_fire"
→ No popup. Greeting guard blocked it.

Bob (ID: 45): "sure sending now"
→ intents: ["money"], conversation_state.status: "fired"
→ triggered_by: {sender: "12", text: "venmo me 20 for lunch", slots: {amount: "20"}}
→ Show Venmo popup to Bob. Pre-fill: pay Alice $20, note "lunch".
```

---

## Intents

| Intent | What it detects |
|--------|-----------------|
| `money` | Venmo/pay/send/owe/split |
| `ride` | Book uber/lyft/cab |
| `food_order` | Order food/delivery |
| `contact` | Call/text/save number |
| `alarm` | Set alarm/wake me |
| `reminder` | Remind me to... / future promise to pay later |
| `calendar` | Schedule/block time |
| `bills` | Pay rent/utilities/electric |
| `travel` | Book flight/hotel |

**MVP scope:** Only `money` and `ride` go through the state machine (request→response flow). Other intents fire immediately as before.

### Slots (flat object)

All slot keys are always returned for the intent — null if not detected. No need to check for key existence.

| Key | Appears with | Example |
|-----|-------------|---------|
| `amount` | money, bills | "30 bucks", "$500" |
| `recipient` | money, contact | "jake", "mom" |
| `note` | money | "dinner", "uber last night" |
| `destination` | ride, travel | "airport", "cancun" |
| `pickup` | ride | "my place" |
| `time` | ride, alarm, reminder, calendar, food_order | "7am", "tomorrow at 3" |
| `food` | food_order | object with food_item, restaurant, etc. |
| `task` | reminder | "call mom", "submit report" |
| `event` | calendar | "team meeting", "dentist" |
| `bill_name` | bills | "electric", "rent" |
| `phone` | contact | "555-1234" |

---

## Popup Cooldown

The server enforces a **30-second cooldown** per (room_id, intent). If money fires at T=0 and another money message arrives at T=10, the second one won't fire. This prevents spam.

You don't need to implement cooldown on your side — the server handles it via `room_id`.

---

## Error Responses

| Status | Meaning |
|--------|---------|
| 400 | `text` is empty |
| 503 | Model still loading (retry after a few seconds) |

---

## WebSocket: `WS /ws/detect`

Alternative to REST if you want persistent connection.

```json
// Send:
{"text": "venmo me 20", "room_id": "dm_12_45", "sender": "12", "context": [{"text": "prev msg", "sender": "45"}]}

// Receive:
{
  "text": "venmo me 20",
  "room_id": "dm_12_45",
  "sender": "12",
  "detection": {
    "intents": [],
    "scores": {...},
    "slots": {...},
    "target": {"show_to": "others", "reason": "..."},
    "money": {...},
    "conversation_state": {"status": "pending", ...},
    "latency_ms": 412.3
  }
}
```

---

## Deployment Notes

- **CPU inference:** ~400-500ms per message
- **GPU (T4/A10G):** ~30-50ms per message
- **Memory:** ~2GB RAM for model
- **Startup:** ~15-20s to load model weights
- **Stateful per room:** The conversation state machine tracks pending requests per room_id in memory. If you scale horizontally, either use sticky sessions or pass `context` with sender info so the server can reconstruct state.
- **Health check:** `GET /health` — wait for `status: "ok"` before routing traffic
