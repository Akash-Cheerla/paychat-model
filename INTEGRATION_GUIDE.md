# PayChat — Money Detection API

So basically I trained a DistilBERT model to pick up money-related messages in chat. Stuff like "you owe me $25", "let's split dinner", "venmo me" etc. The API takes in a chat message and tells you if it's money-related, what kind, how confident it is, and who should get the Venmo popup.

**Model info:**
- DistilBERT (distilbert-base-uncased), fine-tuned on 5,400 examples
- 100% accuracy on test set, 100% F1
- ~300-400ms on CPU, way faster on GPU (~20-50ms)
- Confidence threshold at 0.65 (you can change this)

---

## Getting it running

**With Docker:**
```bash
docker build -t paychat-api .
docker run -p 8000:8000 paychat-api
```

**Without Docker:**
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

**To check everything works:**
```bash
python test_api.py
```
This runs 16 tests against all the endpoints — should all pass green.

---

## How it fits into our flow

```
User sends message in the app
        |
        v
Backend receives it (like it already does)
        |
        v
Backend calls:  POST <API_URL>/detect
                { "text": "you owe me $25", "chat_id": "room123", "sender": "akash" }
        |
        v
My API responds with detection data
        |
        v
Backend attaches that to the message before sending to recipient:
{
  "id": "msg_456",
  "text": "you owe me $25",
  "sender": "akash",
  "timestamp": "...",
  "venmo_detection": {
    "is_money": true,
    "confidence": 0.9979,
    "trigger_type": "owing_debt",
    "direction": "request",
    "detected_amount": "$25"
  }
}
        |
        v
App checks venmo_detection.is_money → if true, show Venmo popup
```

---

## POST /detect

This is the main one. Send every chat message here.

**Request:**
```json
{
  "text": "you owe me $25",
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

Only `text` is required. `chat_id`, `message_id`, `sender` are optional — I just echo them back so you can match responses to messages on your end.

**Response:**
```json
{
  "is_money": true,
  "confidence": 0.9979,
  "trigger_type": "owing_debt",
  "direction": "request",
  "detected_amount": "$25",
  "latency_ms": 342.15,
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

Quick rundown on the fields:
- `is_money` — true/false, the main flag
- `confidence` — 0 to 1, how sure the model is
- `trigger_type` — what category (see below)
- `direction` — **important one** — tells you who should see the popup
- `detected_amount` — pulled out the dollar amount if there is one, null otherwise
- `latency_ms` — how long inference took

Errors: `400` if text is empty, `503` if the model is still loading on cold start.

---

## The direction field

This one matters for the app. It tells you who should actually see the Venmo popup.

| direction | what's happening | who gets the popup |
|-----------|-----------------|-------------------|
| `request` | sender is asking for money ("you owe me $25", "pay me back") | everyone except the sender |
| `offer` | sender wants to pay ("I'll send you $20", "do I owe you?", "shall I send?") | the sender themselves |
| `split` | splitting something ("let's split dinner", "halves?") | everyone |

Here's how the app side should handle it:

```swift
// iOS
guard let det = message.venmo_detection, det.is_money else { return }

switch det.direction {
case "request":
    // they're asking for money, show popup to people who owe
    if currentUser.id != message.sender_id {
        showVenmoPopup(amount: det.detected_amount, to: message.sender)
    }
case "offer":
    // they want to pay, show popup to them
    if currentUser.id == message.sender_id {
        showVenmoPopup(amount: det.detected_amount, to: otherUser)
    }
case "split":
    showVenmoPopup(amount: det.detected_amount, to: message.sender)
default: break
}
```

```kotlin
// Android
message.venmoDetection?.let { det ->
    if (!det.isMoney) return@let
    when (det.direction) {
        "request" -> if (currentUserId != message.senderId) showVenmoPopup(det)
        "offer"   -> if (currentUserId == message.senderId) showVenmoPopup(det)
        "split"   -> showVenmoPopup(det)
    }
}
```

---

## Trigger types

| trigger_type | examples |
|-------------|----------|
| `owing_debt` | "you owe me $25", "pay me back" |
| `bill_splitting` | "let's split dinner", "halves?" |
| `direct_amount` | "that's $50", "30 bucks" |
| `payment_app` | "venmo me", "cashapp me" |
| `general_money` | "cover me", "spot me" |

---

## Other endpoints

**GET /health** — check if the server is up and model is loaded
```json
{
  "status": "healthy",
  "version": { "trained_at": "2026-03-22", "test_accuracy": 1.0, "test_f1": 1.0 },
  "threshold": 0.65
}
```

**GET /metrics** — see how many requests, detections, avg latency
```json
{ "requests": 142, "money_detected": 37, "avg_latency_ms": 312.45 }
```

**POST /reload** — if we retrain the model, call this to swap it in without restarting the server

**WebSocket /ws/detect** — same as POST /detect but over websocket if you prefer that. Send JSON, get JSON back with a `venmo_detection` object attached.

---

## Deployment notes

**Env vars:**
- `MODEL_DIR` — path to the saved_model folder (defaults to `./saved_model`)
- `CONFIDENCE_THRESHOLD` — defaults to 0.65, bump it up if you're getting false positives

**Model files** are in `saved_model/`:
```
config.json, model.safetensors (255 MB), tokenizer.json, tokenizer_config.json, training_report.json
```

The weights file is big (255 MB) so if you're using git, you'll need LFS:
```bash
git lfs install
git lfs track "saved_model/model.safetensors"
```

**For prod** — if you throw this on a GPU instance (like AWS g4dn.xlarge) it'll do inference in 20-50ms instead of 300-400ms on CPU. Use `--workers 1` since the model loads per worker.

Hit `/health` to make sure the model is loaded before sending traffic.

**Docker compose if you need it:**
```yaml
services:
  paychat-detector:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CONFIDENCE_THRESHOLD=0.65
    restart: unless-stopped
```

---

## Retraining

If we ever need to retrain with new data:
```bash
cd training/
pip install -r requirements.txt
python generate_data.py
python train.py
```
Then copy the new `saved_model/` over and hit `POST /reload`.

---

## What it catches vs ignores

It picks up: debt/owing, bill splitting, dollar amounts in payment context, payment app mentions (venmo/cashapp/zelle), general money asks (cover me, spot me, etc.)

It ignores on purpose: stock market talk, prices without payment intent ("that shirt is $50"), news about money, random numbers ("meet at gate 12").

---

## Swagger docs

Once the server is running: `http://<server>:8000/docs`

Hit me up if anything's unclear — Akash
