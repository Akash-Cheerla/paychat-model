# PayChat — Backend Integration


**Models in production (2026-09-23):** DualHeadRoberta **v26** (`saved_model/`) for the nine
intents and slots, plus the **conversation classifier v17** (`conv_model/`, thresholds
0.985 / 0.985) which decides money and ride. v17 went live 2026-09-20.  
**Repo:** https://github.com/Akash-Cheerla/paychat-model

> **Read this before the response-head section below.** Money and ride are NOT decided by the
> response head or the pending store any more; the conversation classifier decides them from
> the last 10 messages. It needs no `reply_to` and no `dm_<lo>_<hi>` room-id convention, and it
> fires in group chats. `conversation_state.decided_by` tells you which path ran, and
> `status` on that path is only ever `fired` or `no_fire`. The other seven intents are
> unchanged. Full detail, measured numbers and rollback: `CONV_CLASSIFIER_DEPLOY.md`.
>
> Two suppression rules were added 2026-09-17, so these produce no prompt: a third person's
> bare "ok"/"sure" after someone already clearly took a group request ("let me book a cab"),
> and a bare yes after a statement that asks for nothing ("im short 300 this month" / "sure").
> Both are in `FIRING_RULE.md` (§1a, §6a).
>
> Keep sending `reply_to` and `participants`: the classifier uses them to put the right
> request's amount and route on the prompt.

---

## What was new in v25 (2026-07, kept for history)

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
| `participants` | int | No | Number of people in the room, **including the sender**. Group rooms only; ignored for `dm_*`. Used to divide a split — see Splits below. Nothing breaks without it. |
| `roster` | object[] | No | Who is in the room, one entry per member **including the sender**: `{"id": "20", "name": "Akash", "nickname": "AK", "places": ["home", "office"]}`. `name` is the first word of the display name, `places` the **names** of the saved places that person has (never coordinates). Lets a ride's "Akash's address" / "your place" resolve to a user id — see `needs_location` below. At most 50 entries; larger or malformed is treated as absent, never rejected. Not stored, not logged, read by nothing else. |
| `client` | string | No | The app build that produced the message, e.g. `ios-1.4.2`. Echoed and logged, never used for classification. |

**`sender` is required now.** Without it the server falls back to immediate-fire mode (no request→response tracking), which defeats the whole point of the state machine.

### Splits — why `participants` matters

"the trip came to 5000, send me your shares" is 1000 each in a room of five and 2500 in
a room of two. The message never says which, and the server has no way to know how many
people are in the room.

- **Headcount stated in the message** ("split 3 ways", "1000 each") — the figure is
  divided, or used as-is if it is already per-person. `participants` is not needed.
- **`participants` supplied** — the total is divided by it.
- **Neither** — the amount comes back **blank** rather than guessed. A payment sheet
  pre-filled with 5000 when the user owes 1000 is worse than one they have to type into.

A split is owed by every member separately, so the request is not consumed by whoever
pays first: each person who commits gets their own prompt for their own share. The same
person restating their commitment does not get a second prompt.

DM rooms infer two people and need nothing.

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
  "needs_location": null,
  "latency_ms": 435.5,
  "chat_id": null,
  "message_id": "msg_789",
  "sender": "12",
  "client": null,
  "model_version": { "base": "v26", "conv": "conv_windows_v17", "decided_by": "conv_classifier" }
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
| `needs_location` | object \| null | Ride pickup/destination the client or server must resolve, and whose they are — see its own section below |
| `model_version` | object | Which models decided (`base`, `conv`, `decided_by`) |
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
>
> **Production runs the conversation classifier (`=1`) as of 2026-08-06** — the right-hand
> column below. Everything documented outside this box describes the **rule layer**, which
> is the code default and the rollback target, but is NOT what your users are hitting.
> Check `conversation_state.decided_by`: it is `"conv_classifier"` on the live path.
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

## `needs_location` — which ride slots to resolve, and whose they are

Present only when a ride is involved and a pickup or destination is self-referential
("my location", "home") or refers to a person ("Akash's address", "your place"). Named
places ("MG Road") get no entry: the client runs those through Places anyway. Emitted on
the **request** as well as on the fire — the rider's app has to know before the fire —
and on a fire the fields are built from `triggered_by.slots`, so `user_id` is
`triggered_by.sender`, the speaker of the phrases, not whoever the prompt opened for.

```json
"needs_location": {
  "user_id": "10",
  "fields": [
    { "slot": "pickup",      "phrase": "My Home",         "resolve": "saved_place", "place": "home", "user_id": "10" },
    { "slot": "destination", "phrase": "Akash's Address", "resolve": "gps",         "place": null,   "user_id": "20" }
  ]
}
```

| Key | Meaning |
|---|---|
| `user_id` | The speaker of the phrases. Kept for clients written before fields had owners. |
| `fields[].slot` | `pickup` or `destination`. |
| `fields[].phrase` | The text as extracted, to show when nothing resolves. |
| `fields[].resolve` | What to do — table below. |
| `fields[].place` | `home`, `office` or null (a current position). |
| `fields[].user_id` | **Whose** location this field is: the speaker for "my …", the matched member for "X's …" / "your …", null when nobody can say. |
| `fields[].name` | Only on an unresolved person phrase: the name as written, for the log. |
| `fields[].who` | `"other_party"` on a "your …" phrase that could not be resolved (no roster, or a group). |

| `resolve` | When | Expected handling |
|---|---|---|
| `gps` | "my location", "here", or "X's current location" with exactly one roster match | The server asks `user_id`'s device over its socket if that user is someone else; the device owner's own client fills from GPS |
| `saved_place` | "my home", "office", or "X's home" **when that person's `places` has it** | Looked up from the saved-places table for `user_id`, coordinates written into the slot |
| `ask` | "my hostel", a name matching nobody or more than one member, a saved place the person has not stored, "his/their …", or "your …" in a group | Open the address picker with `phrase` pre-filled |
| `participant` | Only when **no roster was sent**: a person phrase with `name` (and `who` for "your …") | Whoever holds the roster matches it; treat as `ask` otherwise |

Resolution rules: a name matches a roster entry when it equals that entry's `name` or
`nickname` case-insensitively — exact token, one match only. "your …" resolves to the other
entry in a two-person roster (a DM) and is `ask` in a group. Curly apostrophes are
normalised, and "address"/"addr" read as home. Every rule above degrades to `ask` rather
than guess: a wrong person's home in a cab booking is worse than a picker.

## Popup Cooldown

There is **no cooldown** on `/classify`. An earlier server enforced one per (room, intent);
the conversation state machine replaced it, and the only thing that still applies one is
the `/ws/{room}/{user}` demo endpoint. Do not implement one client-side either.

---

## Endpoint: `GET /room-summary/{room_id}` — what's open, what was prompted

Added 2026-09-27 for live traffic. Built from the state `/classify` already keeps, so it
describes the same conversation the classifier sees. Read-only: calling it cannot change a
later prompt.

```
GET /room-summary/dm_10_20            # everything in the room
GET /room-summary/dm_10_20?sender=20  # what user 20 has to act on
```

With `sender`, it answers "what do I have to act on": `prompts` holds only the ones aimed at
that user, `open_requests` only what **other** people asked, and their own unanswered asks move
to `my_requests` — sorted out, not hidden. Without `sender`, every open request is listed,
because in a group one may be waiting for anybody.

```json
{
  "room": "dm_10_20",
  "messages_seen": 14,
  "open_requests": [
    { "intent": "ride", "from": "10", "text": "can someone book me a cab to the airport",
      "message_id": "m_41", "slots": { "destination": "Airport" },
      "divisible": true, "answered_by": [], "age_seconds": 92 }
  ],
  "prompts": [
    { "ts": "2026-09-27T09:14:03Z", "intent": "money", "to": "20", "said_by": "20",
      "message_id": "m_44", "text": "sure", "slots": { "amount": "$500", "note": "lunch" },
      "answers": { "from": "10", "text": "can you send me 500 for lunch", "message_id": "m_39" } }
  ],
  "settled": { "money": 310 },
  "window_ttl_seconds": 14400,
  "as_of": "2026-09-27T09:15:35Z"
}
```

**Every timestamp here is UTC with the `Z`.** Changed 2026-09-30, after Brahma asked what
timezone `ts` was in: the old form (`2026-09-27T09:14:03`, no offset) is read as LOCAL time
by JavaScript's `Date` — 5h30m out on an IST device, silently — rejected by Swift's
`ISO8601DateFormatter`, and thrown on by Kotlin's `Instant.parse`. `ts`, `as_of`, and
`started_at` / `loaded_at` on `/health` all carry it now. For "how old is this", prefer
`age_seconds` on a request: it is an int computed here, with no timezone in it at all.

| field | meaning |
|---|---|
| `open_requests` | asked, nobody has answered yet. An ordinary request leaves the list when someone takes it; a split (`divisible: true`) stays, and `answered_by` lists the user ids who have already paid their share |
| `my_requests` | only with `sender`: that user's own unanswered asks |
| `prompts` | prompts this server showed, oldest first, last 25 per room. `to` is who performs the action (the payer or the booker) — the same person `target.show_to` points at. `answers` is the request it carried |
| `expired` | on a request the classifier can no longer see (older than 4h). It is still listed, because hiding it would make "nothing was asked" and "it aged out" look identical — but nobody can answer it any more |
| `settled` | seconds since the last completed prompt per intent |
| `messages_seen` | messages this room has sent through `/classify` since the server started |

**Why 4 hours.** That is the classifier's memory (`PAYCHAT_CONV_CONTEXT_TTL`), not a choice this
endpoint makes: money and ride are decided by reading the conversation window, so a request
older than the window cannot be answered by anyone. Five minutes was once too short — a reply
seven minutes after the request scored 0.03 instead of 0.997 — and a day is too long, because a
stale request attaches itself to an unrelated "sure". The value is configurable, and
`window_ttl_seconds` in the response says what the server is running.

`prompts` is **not** age-capped: it is the last 25 for the room, however old. One limit applies
to all of it: **memory only**, so a restart empties this exactly as it empties the conversation
window. With the classifier off the endpoint returns an `error` string rather than pretending to
know.

**Not** `/summary/{room_id}/{user_name}`: that one only ever sees the demo WebSocket chat
(`/chat`) and returns empty for real rooms.

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
