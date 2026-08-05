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

This is the only endpoint you need. Call it for every message (DMs and group chats).

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
| `room_id` | string | **Yes** | Room ID. DMs: `dm_<lo>_<hi>` format. Groups: any other format (e.g. `group_abc`). The prefix determines matching behavior — see DM vs Group below. |
| `sender` | string | **Yes** | User ID of whoever sent this message. The state machine uses this to know who's responding to whose request. |
| `context` | object[] | No | Previous messages. Each object: `{"text": "...", "sender": "..."}`. Pass last 2-3 messages. If omitted, server tracks internally per room_id. |
| `message_id` | string | No | Message ID (echoed back, also stored with pending requests for `triggered_by`) |
| `reply_to` | string | No | Message ID of the message being replied to (from the chat app's reply-to-message feature). **Required for group chats** — see below. |

**`sender` is required now.** Without it the server falls back to immediate-fire mode (no request→response tracking), which defeats the whole point of the state machine.

**`context` format changed.** Old format was a plain string array — that still works for backward compat but you lose sender info on context messages. New format is an array of `{"text": "...", "sender": "..."}` objects. Sender on context messages matters because the state machine needs to know who said what.

### DM vs Group Chat Matching

This is important. The state machine matches responses to pending requests differently depending on room type:

**DMs (`dm_*` rooms):** Ambient matching — no `reply_to` needed. Only two people in the room, so when one person requests and the other responds, it's always unambiguous. This is the behavior described in all the examples above.

**Groups (any room not starting with `dm_`):** `reply_to` is **required** to match a response. Without it, responses are ignored for money/ride. This prevents a problem: if Priya asks Liam for $15 and Maya asks Jake for $45 in the same group, Jake's "bet" shouldn't accidentally match Priya's request.

```json
// Group chat — Priya requests
{"text": "liam venmo me 15", "room_id": "group_squad", "sender": "priya", "message_id": "msg_1"}
// → status: "pending"

// Liam swipe-replies to Priya's message
{"text": "bet sending now", "room_id": "group_squad", "sender": "liam", "reply_to": "msg_1"}
// → status: "fired", triggered_by.sender: "priya"

// Jake says "bet" without replying to anyone (no reply_to)
{"text": "bet", "room_id": "group_squad", "sender": "jake"}
// → no match, no fire — safe
```

**Expired requests:** Pending requests expire after 5 minutes or 10 messages from others. But expired requests are **archived for 48 hours**. If someone replies to a money request the next day using the chat app's reply feature, `reply_to` matches against the archive and still fires.

**How to pass `reply_to`:** When a user swipe-replies (or long-press replies) to a message, pass that original message's ID as `reply_to`. Most chat frameworks already expose this — WhatsApp, Telegram, iMessage all have it. Just pass it through.

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
| `intents` | string[] | Fired intents (empty if nothing detected or if request is stored as pending). **Only `money` and `ride` are surfaced** — see below |
| `scores` | object | Confidence per intent (0.0–1.0) |
| `slots` | object \| null | Extracted entities (flat key-value) |
| `money` | object \| null | Money enrichment if money intent fired |
| `target` | object \| null | Who should see the popup |
| `conversation_state` | object \| null | State machine result — see below |
| `lifecycle` | object \| null | Cancel/defer/confirm state changes |
| `guardrails` | object \| null | Compliance flags (PCI, AML, phishing) |
| `latency_ms` | float | Inference time |

### Which intents are surfaced

As of 2026-08-05 the server returns **`money` and `ride` only**. The model still scores
all nine and `scores` still contains all nine — that is what the dogfood logs capture —
but the other seven never appear in `intents`, so no client can act on them.

They were split out of the training data months ago and never retrained, so they misfire
on ordinary chat. Each comes back once it has had its own training round.

```bash
# default — money and ride only
PAYCHAT_ACTIVE_INTENTS=money,ride

# widen selectively as intents are retrained
PAYCHAT_ACTIVE_INTENTS=money,ride,contact

# all nine, pre-2026-08-05 behaviour
PAYCHAT_ACTIVE_INTENTS=all
```

The server refuses to start on an unrecognised intent name, so a typo fails loudly
rather than silently disabling everything.

---

## Conversation State Machine (the big change)

For money and ride intents, the model doesn't fire immediately anymore. It works in two steps:

1. **Request** — "venmo me $20" → intent detected but stored as **pending**, not fired. `intents` comes back empty.
2. **Response** — "sure, sending now" → state machine detects this as an ack to the pending request → money intent **fires** on this message.

This prevents false positives. Someone saying "venmo me $20" isn't an action yet — it's a request. The action happens when someone responds.

### `conversation_state` field

Present on every response. Tells you what the decision layer decided.

> #### ⚠️ Two decision layers — check `decided_by`
>
> The server can decide money/ride two ways, selected by `PAYCHAT_CONV_CLASSIFIER`.
> Everything documented below describes the **rule layer**, which is what runs today.
>
> | | rule layer (default) | conversation classifier (`=1`) |
> |---|---|---|
> | `decided_by` | absent | `"conv_classifier"` |
> | `status` values | `pending`, `fired`, `reminder`, `cancelled`, `no_fire` | **only `fired` and `no_fire`** |
> | `triggered_by` | on fire | on fire — same shape, same slots |
> | groups | need `reply_to`, else nothing fires | fire without `reply_to` |
> | `room_id` must start `dm_` | yes, for ambient matching | no |
>
> **What still works unchanged:** gate the payment prompt on `status == "fired"` and read
> the amount from `triggered_by.slots`. Both behave identically on either path.
>
> **What goes quiet:** `pending`, `reminder` and `cancelled` are rule-layer states the
> classifier does not model. A request produces `no_fire` until someone commits; a
> deferral or a rejection also produces `no_fire`. Any handler branching on those three
> stops being reached — gate them on `decided_by` being absent if you need both paths.
>
> **What starts happening:** group chats fire. Today a group needs the responder to
> swipe-reply; the classifier reads the last 10 messages and does not. If prompts appear
> in groups after the flag is switched on, that is intended.
>
> One case fires nothing on either path: two different requests open at once answered
> with a bare "ok". There is no signal for which one is meant. Naming the action
> ("ok sending" / "ok booking") resolves it.
>
> Full deploy notes, measured numbers and known gaps: `CONV_CLASSIFIER_DEPLOY.md`.

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
