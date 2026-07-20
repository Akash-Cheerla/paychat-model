# PayChat v25 — Integration Guide

## Overview

PayChat detects 9 actionable intents in chat messages (money, ride, food_order, etc.) and extracts slots (recipient, amount, time, destination). It runs as an HTTP API with a conversation state machine that tracks pending requests and fires intents on responses.

**Model:** DualHeadRoberta v25 — RoBERTa-base with 4 heads (topic, action, projection, response)  
**Latency:** 30-50ms (GPU), 400-500ms (CPU)  
**Response head accuracy:** 92.7% on held-out test set

---

## Architecture

```
User message
    │
    ▼
┌─────────────────────────────────────┐
│  POST /classify                     │
│                                     │
│  Phase 1: RoBERTa inference         │
│    → 9-intent scores + CLS embed    │
│                                     │
│  Phase 1b: Money enrichment         │
│    → amount, direction, trigger     │
│                                     │
│  Phase 2: Target detection          │
│    → show_to: sender/others/group   │
│                                     │
│  Phase 3: Slot extraction           │
│    → amount, recipient, time, etc.  │
│                                     │
│  Phase 4: Conversation state machine│
│    → pending/fired/reminder/cancel  │
│    → response head classifies acks  │
│                                     │
│  Phase 5: Lifecycle (cancel/defer)  │
│  Phase 6: Guardrails (PCI/AML)     │
└─────────────────────────────────────┘
    │
    ▼
{ intents, slots, target, conversation_state, ... }
```

### State Machine Flow (money + ride only)

**DMs** — ambient matching, no `reply_to` needed:
```
Alice: "venmo me $20"
  → Model detects money intent
  → State machine stores as PENDING (intents: [])
  → No popup shown

Bob: "sure sending now"  
  → Response head classifies as positive_ack
  → State machine FIRES money intent
  → intents: ["money"], triggered_by: {sender: "Alice", ...}
  → Show Venmo popup to Bob
```

**Groups** — requires `reply_to` (the original message ID) to match:
```
Priya: "liam venmo me 15"  (message_id: "msg_1")
  → Stored as PENDING

Jake: "bet"  (no reply_to)
  → No match — can't know which request this is for

Liam: "bet sending now"  (reply_to: "msg_1")
  → Matches Priya's request → FIRES
  → triggered_by: {sender: "Priya", ...}
```

Expired requests are archived for 48 hours — `reply_to` works even on next-day responses.

Other intents (food, alarm, calendar, etc.) fire immediately — no state machine.

---

## E2EE Data Flow

The backend/database never touches plaintext. Context comes from the client.

```
┌──────────┐       POST /classify        ┌─────────────────────┐
│  Client   │ ──────────────────────────► │  Model Server       │
│           │   {text, context, sender,   │                     │
│ • decrypts│    room_id}                 │ • processes in RAM  │
│   messages│                             │ • state machine     │
│   for     │ ◄────────────────────────── │   tracks pending    │
│   display │   {intents, slots, target,  │ • nothing logged    │
│           │    conversation_state}      │                     │
└──────────┘                              └─────────────────────┘
     │                                           ▲
     │  encrypted messages                       │
     ▼                                           │
┌──────────┐                                     │
│ Backend  │  ← never sees plaintext             │
│ (Elixir) │  ← relays ciphertext               │
│          │  ← calls /classify with decrypted   │
└──────────┘    text from client                 │
```

---

## API Docs

- **Backend team (Phoenix/Elixir):** See [API_BACKEND.md](API_BACKEND.md) — full `/classify` endpoint spec, conversation_state reference, `triggered_by` field.
- **Mobile team (iOS/Android):** See [API_MOBILE.md](API_MOBILE.md) — what the detection payload looks like, when to show popups, Swift/Kotlin examples.

---

## Running Locally

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
MODEL_DIR=./saved_model python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Or with Docker:
```bash
docker build -t paychat .
docker run -p 8000:8000 paychat
```

---

## Deployment

- Python 3.10+, PyTorch 2.0+, transformers, fastapi, uvicorn, spacy
- ~2GB RAM for model
- GPU optional (faster but not required)
- Health check: `GET /health` — wait for `status: "ok"` before routing traffic
- State machine is in-memory per room_id. For horizontal scaling, use sticky sessions or pass full `context` with sender info.

---

## Files

| File | What it does |
|------|-------------|
| `app.py` | FastAPI server — all endpoints, model loading, full pipeline |
| `conversation.py` | Conversation state machine — pending requests, response classification, `triggered_by` |
| `saved_model/` | Model weights, tokenizer, thresholds |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker build |
