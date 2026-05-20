# FYOE Intent Model — Integration Guide

## Overview

The FYOE intent model classifies chat messages into 18 actionable intents (ride, food_order, money, alarm, etc.) and extracts slots (recipient, amount, time, destination, etc.). It runs as a **stateless HTTP API** — no message storage, no logs, no persistent state.

**Model**: Fine-tuned RoBERTa-base (125M params, ~500MB)  
**Latency**: 20-40ms per message (GPU), 80-150ms (CPU)  
**Accuracy**: 98.4% on 243 real-world tests, 100% on seed suite  

---

## API Spec

### `POST /classify`

Classify a single message with optional conversation context.

**Request:**
```json
{
  "text": "yeah dominos",
  "context": ["what should we eat tonight"],
  "room_id": "room_abc123"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | yes | Current message to classify |
| `context` | string[] | no | Last 1-2 prior messages (decrypted client-side). See Context Rules below. |
| `room_id` | string | no | Room identifier for logging (not stored) |

**Response:**
```json
{
  "fired": ["food_order"],
  "slots": {
    "food_order": {
      "item": "dominos",
      "provider": "dominos"
    }
  },
  "needs_clarification": [],
  "scores": [
    {"intent": "food_order", "prob": 0.92, "fired": true},
    {"intent": "ride", "prob": 0.03, "fired": false}
  ],
  "should_popup": ["food_order"],
  "ui_active": ["food_order"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `fired` | string[] | Intents that passed threshold (can be multiple) |
| `slots` | object | Extracted slot values per intent |
| `needs_clarification` | object[] | Missing required slots (app should ask user) |
| `scores` | object[] | All 18 intent probabilities (for debugging) |
| `should_popup` | string[] | **Event** — fire a new popup animation. Respects 30s cooldown so the same intent doesn't re-animate on every message in a conversation. |
| `ui_active` | string[] | **State** — keep the action button visible. Always matches `fired`. Client owns dismissal (user completes action, taps away, or times out). |

> **UI pattern**: on each message, use `should_popup` to trigger the animation and `ui_active` to decide if the button stays rendered. They diverge when the same intent re-fires within 30s — you skip the animation but keep the button up.

### `POST /batch`

Classify multiple messages at once (for catch-up / summary flows).

**Request:**
```json
{
  "messages": [
    {"text": "uber to the airport", "context": []},
    {"text": "venmo me 30", "context": []}
  ]
}
```

---

## 18 Intents

| Intent | Example | Slots |
|--------|---------|-------|
| `money` | "venmo me 30 for pizza" | amount, recipient, note, method, direction |
| `alarm` | "remind me at 6pm" | datetime, note, recurrence |
| `contact` | "call mom" | name, channel, note |
| `calendar` | "meeting at 3pm tomorrow" | title, datetime, with_who, location |
| `maps` | "how do i get to the airport" | destination, origin, mode |
| `ride` | "uber to the mall" | destination, pickup, when, service |
| `food_order` | "order chipotle" | item, provider, when, address |
| `travel` | "flights to bali in december" | destination, origin, when, mode |
| `shopping` | "buy new headphones" | item, qty, provider, when |
| `music` | "play some drake" | song, artist, provider |
| `video` | "watch succession tonight" | title, provider |
| `tickets` | "concert tickets for travis scott" | event, when, qty |
| `reservation` | "book a table at nobu" | place, when, qty |
| `task` | "finish the report" | action, when |
| `note` | "save this for later" | body |
| `bills` | "pay the electric bill" | bill_kind, amount, due |
| `health` | "schedule a dentist appointment" | provider, when, kind |
| `weather` | "is it gonna rain today" | place, when |

---

## Context (Multi-Turn)

The model understands conversation context — "yeah" after "order sushi?" correctly fires `food_order`, and "nah im good" after the same message fires nothing.

### E2EE Data Flow

**The backend/database never touches plaintext.** Context comes from the client, which already decrypts messages for display.

```
┌──────────┐       POST /classify        ┌─────────────────────┐
│  Client   │ ──────────────────────────► │  Model Server       │
│           │   {text, context, room_id}  │  (Nitro Enclave)    │
│ • decrypts│                             │                     │
│   messages│ ◄────────────────────────── │ • processes in RAM  │
│   for     │   {fired, slots,           │ • nothing stored    │
│   display │    should_popup, ui_active} │ • garbage collected │
└──────────┘                              └─────────────────────┘
     │                                           ▲
     │  encrypted messages                       │
     ▼                                           │
┌──────────┐                                     │
│ Backend  │  ← never sees plaintext             │
│ (E2EE)   │  ← no decryption keys              │
│          │  ← just stores/relays ciphertext    │
└──────────┘                                     │
                    vsock (enclave) ──────────────┘
```

**Step by step:**
1. User sends encrypted message → backend relays to recipient
2. Client decrypts message locally (already doing this for display)
3. Client calls `POST /classify` with decrypted text + last 1-2 messages it already has in memory
4. Model server processes in RAM inside the enclave → returns intent + slots
5. Response goes back to client → client shows popup/action button
6. Nothing is stored, nothing is logged — RAM only, garbage collected after response

### How to send context:
Just pass the last 1-2 messages from the chat room as the `context` array. The server handles all the smart filtering internally:
- Rejections ("nah", "nvm", "im good") → server drops context automatically
- Filler ("lol", "nice", "haha") → server drops context automatically
- Confirmations ("yeah", "bet", "sure") → server uses context to resolve intent

```json
{
  "text": "yeah lets do it",
  "context": ["wanna order sushi?"],
  "room_id": "room_abc123"
}
```

**Client just needs to:**
1. Decrypt messages locally (already doing this for display)
2. Send last 1-2 messages as `context` — no filtering needed on your end
3. Pass a consistent `room_id` so the server can track popup cooldowns

The server is stateless — nothing is stored or logged.

---

## Deployment

### Requirements:
- Python 3.10+
- PyTorch 2.0+
- transformers
- fastapi + uvicorn
- ~2GB RAM (model + inference)
- GPU optional (faster but not required)

### Docker (for Nitro Enclave):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY eval/ eval/
COPY saved_model/ saved_model/
ENV MODEL_DIR=/app/saved_model
EXPOSE 8001
CMD ["python", "eval/eval_server.py"]
```

### Environment:
- `MODEL_DIR`: path to saved_model/ (default: `/app/saved_model` in Docker, `../saved_model` locally)
- `PORT`: server port (default: 8001)
- No persistent storage needed
- No logging of message content
- Memory-only operation — fits enclave constraints

---

## Security Notes

- Server is **stateless** — no message storage, no conversation logs
- Context accumulator runs in-memory only, auto-expires after 30 min inactivity
- No plaintext is written to disk at any point
- For Nitro Enclave: model weights are the only file read from disk
- All message content stays in RAM during inference, garbage collected after response

---

## Reference Implementation

`eval/eval_server.py` is the working reference:
- WebSocket chat interface with real-time classification
- REST endpoints for batch processing
- Context accumulator integrated
- Slot extraction + clarification detection
- Smart post-processing (12 rules to kill false positives)

To run locally:
```bash
cd paychat-model
pip install torch transformers fastapi uvicorn python-dateutil
python eval/eval_server.py
# → http://localhost:8001
```
