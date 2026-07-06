# PayChat v21 — Backend Integration

**For:** Samyak (Phoenix/Elixir backend)  
**Model:** DualHeadRoberta v21 (9 intents, context-aware)  
**Repo:** https://github.com/Akash-Cheerla/paychat-model

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
  "sender": "alice",
  "context": ["hey are you free tonight?", "yeah lets grab food"],
  "message_id": "msg_789"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | **Yes** | The message to classify |
| `room_id` | string | No | DM room ID (enables conversation tracking + popup cooldown) |
| `sender` | string | No | Sender identifier (echoed back) |
| `context` | string[] | No | Previous messages in the conversation (max 3 used). If omitted, server uses its own internal tracking per room_id |
| `message_id` | string | No | Message ID (echoed back) |

**About `context`:** Pass the last few messages from the conversation as a plain string array — each element is just the message text, not a message object (e.g. `["hey are you free?", "yeah lets grab food"]`). The server uses up to 3 most recent ones to understand conversation flow. If you don't pass `context`, the server tracks messages internally per `room_id` — but passing it explicitly is recommended since Phoenix already has the messages.

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
  "lifecycle": null,
  "guardrails": null,
  "context_boosted": null,
  "latency_ms": 435.5,
  "chat_id": null,
  "message_id": "msg_789",
  "sender": "alice"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `intents` | string[] | Fired intents (empty if nothing detected) |
| `scores` | object | Confidence per intent (0.0–1.0) |
| `slots` | object \| null | Extracted entities (flat key-value) |
| `money` | object \| null | Money enrichment if money intent fired |
| `target` | object \| null | Who should see the popup |
| `lifecycle` | object \| null | Cancel/defer/confirm state changes |
| `guardrails` | object \| null | Compliance flags (PCI, AML, phishing) |
| `latency_ms` | float | Inference time |

### `target.show_to` — Who Gets the Popup

| Value | Meaning | Example |
|-------|---------|---------|
| `"sender"` | Show popup to the person who sent the message | "book me an uber", "order pizza" |
| `"others"` | Show popup to everyone except sender | "venmo me 30", "you owe me" |
| `"group"` | Show to everyone | "let's split dinner" |

### Intents

| Intent | What it detects |
|--------|-----------------|
| `money` | Venmo/pay/send/owe/split |
| `ride` | Book uber/lyft/cab |
| `food_order` | Order food/delivery |
| `contact` | Call/text/save number |
| `alarm` | Set alarm/wake me |
| `reminder` | Remind me to... |
| `calendar` | Schedule/block time |
| `bills` | Pay rent/utilities/electric |
| `travel` | Book flight/hotel |

### Slots (flat object)

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

The server enforces a **30-second cooldown** per (room_id, intent). If money fires at T=0 and another money message arrives at T=10, the second one won't be in `should_popup`. This prevents spam.

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
{"text": "venmo me 20", "room_id": "dm_12_45", "sender": "alice", "context": ["prev msg"]}

// Receive:
{
  "text": "venmo me 20",
  "room_id": "dm_12_45",
  "sender": "alice",
  "detection": {
    "intents": ["money"],
    "scores": {...},
    "slots": {...},
    "target": {"show_to": "others", "reason": "..."},
    "money": {...},
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
- **Stateless:** Can scale horizontally. Conversation context can be passed via `context` field so no sticky sessions needed.
- **Health check:** `GET /health` — wait for `status: "ok"` before routing traffic
