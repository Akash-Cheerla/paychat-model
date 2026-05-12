# FYOE Eval Server API — v5

Base URL: `http://localhost:8001`

This is the **eval/testing server**, not the production API. In prod everything runs in the Nitro Enclave. This server lets you stress-test the model by throwing messages at it and seeing what comes back.

---

## Model quick facts

- RoBERTa-base, 18 sigmoid heads (multi-label)
- ~30,840 training examples (2,486 multi-turn context), 8 epochs
- Focal loss + label smoothing + fp16 mixed precision
- Per-intent optimized thresholds (not flat 0.5)
- **v5: multi-turn context** — model sees last 3-5 messages via tokenizer pair encoding
- Layer A slot filler: regex + dateparser, <5ms/message, $0
- Trigger filter: regex + fuzzy Levenshtein (catches typos automatically)

### The 18 intents

`money`, `alarm`, `contact`, `calendar`, `maps`, `food_order`, `ride`, `travel`, `shopping`, `music`, `video`, `tickets`, `reservation`, `task`, `note`, `bills`, `health`, `weather`

---

## `POST /detect`

The main one. Send a message, get back all 18 scores + slots.

**Request:**
```json
{ "text": "remind me to venmo priya $25 at 8pm" }
```

**Response:**
```json
{
  "text": "remind me to venmo priya $25 at 8pm",
  "all_scores": [
    { "intent": "alarm",    "prob": 0.9912, "threshold": 0.60, "fired": true },
    { "intent": "money",    "prob": 0.9847, "threshold": 0.30, "fired": true },
    { "intent": "calendar", "prob": 0.0312, "threshold": 0.29, "fired": false }
  ],
  "fired": ["money", "alarm"],
  "slots": {
    "money": {
      "amount": "25",
      "recipient": "priya",
      "note": null,
      "method": "venmo",
      "direction": "send",
      "_required_filled": true
    },
    "alarm": {
      "datetime": "2026-05-12T20:00:00",
      "note": "venmo priya $25",
      "_required_filled": true
    }
  },
  "needs_clarification": []
}
```

`all_scores` is sorted by probability (highest first). `fired` is the list of intents that cleared their threshold. `slots` has per-intent extracted entities. `needs_clarification` tells you which required slots are missing.

---

## `POST /batch`

Same thing but for a list of messages at once.

```json
{ "texts": ["you owe me $20", "remind me at 6pm", "order pizza"] }
```

Returns `{ "results": [ ...same shape as /detect... ] }`.

---

## `GET /meta`

Model version, all 18 labels, thresholds, slot schema.

---

## `POST /judge`

Save a human verdict for a tested message. The eval UI calls this when you click correct/wrong.

```json
{
  "text": "call dad",
  "predicted": ["contact"],
  "expected": ["contact"],
  "verdict": "correct",
  "note": "",
  "tag": "contact-family"
}
```

`verdict`: `correct` | `wrong_intent` | `missing_intent` | `extra_intent` | `wrong_entity`

---

## `GET /stats`

Running accuracy from saved judgments — total, today, per-intent precision/recall, by tag, recent failures.

---

## `GET /failures`

All judgments where verdict != correct.

---

## `GET /export`

Failures as training-data JSONL. Drop into `training/data/` for the next retrain.

---

## `GET /seed`

The 126 adversarial seed-suite test cases as JSON (including 16 multi-turn context tests).

---

## WebSocket `/ws/{room_id}/{user_name}`

Powers the two-person chat. Connect, send messages, get back model predictions broadcast to everyone in the room.

**v5:** The WebSocket chat now uses the **context accumulator** — every message is classified with the last 5 messages from the room as context. This means confirmations ("yeah do it"), context buildup ("I'm hungry" → "dominos?"), and corrections ("actually make it lyft") all work.

**Connect:** `ws://localhost:8001/ws/myroom/akash`

**Send:** `{ "type": "msg", "text": "venmo priya $25" }`

**Receive:**
```json
{
  "type": "msg",
  "sender": "akash",
  "text": "venmo priya $25",
  "fired": ["money"],
  "slots": { "money": { "amount": "25", "recipient": "priya" } },
  "needs_clarification": [],
  "context_used": true,
  "top_scores": [ ... ],
  "ts": "2026-05-12T10:30:00+00:00"
}
```

`context_used: true` means prior messages from the room were used as context for this classification.

Other frames: `history` (on connect — last 50 msgs), `presence` (join/leave), `ping`/`pong` (keepalive).

---

## Per-intent thresholds (v5)

Each intent has its own threshold from validation. Not a flat 0.5.

| intent | threshold | intent | threshold |
|---|---|---|---|
| money | 0.30 | shopping | 0.20 |
| alarm | 0.60 | music | 0.29 |
| contact | 0.45 | video | 0.29 |
| calendar | 0.29 | tickets | 0.36 |
| maps | 0.66 | reservation | 0.48 |
| food_order | 0.29 | task | 0.64 |
| ride | 0.57 | note | 0.16 |
| travel | 0.17 | bills | 0.64 |
| | | health | 0.20 |
| | | weather | 0.14 |

Baked into `saved_model/thresholds.json`, loaded automatically.
