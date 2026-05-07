# FYOE Eval Server API — v4

Base URL: `http://localhost:8001`

This is the **eval/testing server**, not the production API. In prod everything runs on-device (E2EE). This server lets you stress-test the model by throwing messages at it and seeing what comes back.

---

## Model quick facts

- RoBERTa-base, 18 sigmoid heads (multi-label)
- 19,001 training examples, 5 epochs
- Per-intent optimized thresholds (not flat 0.5 — each intent has its own cutoff)
- Layer A slot filler: regex + dateparser, <5ms/message, $0
- 98.8% test exact match, 81.7% adversarial seed suite

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
    { "intent": "alarm",    "prob": 0.9912, "threshold": 0.30, "fired": true },
    { "intent": "money",    "prob": 0.9847, "threshold": 0.55, "fired": true },
    { "intent": "calendar", "prob": 0.0312, "threshold": 0.70, "fired": false },
    ...
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
      "datetime": "2026-05-02T20:00:00",
      "note": "venmo priya $25",
      "_required_filled": true
    }
  },
  "needs_clarification": []
}
```

`all_scores` is sorted by probability (highest first). `fired` is the list of intents that cleared their threshold. `slots` has per-intent extracted entities. `needs_clarification` tells you which required slots are missing — if it's not empty, the app should ask the user instead of guessing.

---

## `POST /batch`

Same thing but for a list of messages at once.

```json
{ "texts": ["you owe me $20", "remind me at 6pm", "order pizza"] }
```

Returns `{ "results": [ ...same shape as /detect... ] }`.

---

## `GET /meta`

Model version, all 18 labels, thresholds, slot schema. Good for sanity-checking what the server is running.

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

The 82 adversarial seed-suite test cases as JSON.

---

## WebSocket `/ws/{room_id}/{user_name}`

Powers the two-person chat. Connect, send messages, get back model predictions broadcast to everyone in the room.

**Connect:** `ws://localhost:8001/ws/myroom/akash`

**Send:** `{ "type": "msg", "text": "venmo priya $25" }`

**Receive:**
```json
{
  "type": "msg",
  "sender": "akash",
  "text": "venmo priya $25",
  "fired": ["money"],
  "slots": { "money": { "amount": "25", "recipient": "priya", ... } },
  "needs_clarification": [],
  "top_scores": [ ... ],
  "ts": "2026-05-02T10:30:00+00:00"
}
```

Other frames: `history` (on connect — last 50 msgs), `presence` (join/leave), `ping`/`pong` (keepalive).

---

## Per-intent thresholds

Each intent has its own threshold from validation. Not a flat 0.5.

| intent | threshold | intent | threshold |
|---|---|---|---|
| money | 0.55 | shopping | 0.10 |
| alarm | 0.30 | music | 0.30 |
| contact | 0.20 | video | 0.35 |
| calendar | 0.70 | tickets | 0.10 |
| maps | 0.20 | reservation | 0.20 |
| food_order | 0.20 | task | 0.70 |
| ride | 0.50 | note | 0.10 |
| travel | 0.10 | bills | 0.20 |
| | | health | 0.45 |
| | | weather | 0.10 |

Baked into `saved_model/thresholds.json`, loaded automatically.
