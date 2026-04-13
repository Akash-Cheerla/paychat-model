# PayChat Money Detection API

Base URL: `http://<your-server>:8000`

---

## `POST /detect`

Send a chat message, get back whether it's about money and what to do with it.

### Request

```json
{
  "text": "you owe me $20 from last night",
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
  "is_money": true,
  "confidence": 0.9847,
  "trigger_type": "owing_debt",
  "direction": "request",
  "detected_amount": "$20",
  "latency_ms": 245.3,
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `is_money` | boolean | `true` if the message is about money/payments |
| `confidence` | float | Model confidence (0.0 to 1.0). Threshold is 0.65 |
| `trigger_type` | string or null | Why it triggered. One of: `owing_debt`, `bill_splitting`, `direct_amount`, `payment_app`, `general_money`. Null if `is_money` is false |
| `direction` | string or null | Who should see the popup. One of: `request`, `offer`, `split`. Null if `is_money` is false |
| `detected_amount` | string or null | Extracted dollar amount (e.g. `"$20"`, `"$15.50"`). Null if no amount found |
| `latency_ms` | float | How long inference took in milliseconds |
| `chat_id` | string or null | Passed through from request |
| `message_id` | string or null | Passed through from request |
| `sender` | string or null | Passed through from request |

### Direction field — who gets the popup

| Direction | Meaning | Show popup to |
|-----------|---------|--------------|
| `request` | Sender is asking for money ("you owe me $20") | Recipients (everyone except sender) |
| `offer` | Sender is offering to pay ("I'll venmo you") | Sender |
| `split` | Mutual split ("let's split the bill") | Everyone in the chat |

### Trigger types

| Trigger Type | Example Messages |
|-------------|-----------------|
| `owing_debt` | "you owe me $20", "pay me back", "where's my money" |
| `bill_splitting` | "let's split the bill", "your share is $15", "go halves?" |
| `direct_amount` | "send me $50", "that'll be $30", "I need $20 from you" |
| `payment_app` | "venmo me", "cashapp me $20", "zelle me" |
| `general_money` | "my treat", "I got you", "drinks on me", "spot me" |

### Error responses

| Status | Body | When |
|--------|------|------|
| 400 | `{"detail": "text cannot be empty"}` | Empty or missing text |
| 503 | `{"detail": "Model not loaded..."}` | Server still starting up (cold start ~20s) |

---

## `GET /health`

Health check. Use this for your load balancer or uptime monitoring.

```json
{
  "status": "healthy",
  "device": "cpu",
  "model_dir": "/app/saved_model",
  "loaded_at": "2026-04-11T10:30:00",
  "version": {
    "trained_at": "2026-04-12T...",
    "test_accuracy": 1.0,
    "test_f1": 1.0
  },
  "threshold": 0.65,
  "total_requests": 1284
}
```

---

## `GET /metrics`

Live stats. Good for dashboards.

```json
{
  "requests": 1284,
  "money_detected": 342,
  "detection_rate": 0.2664,
  "avg_latency_ms": 231.5,
  "started_at": "2026-04-11T10:30:00"
}
```

---

## `POST /reload`

Hot-reload the model without restarting the server. Call this after you update the model files on disk.

```json
{
  "status": "ok",
  "loaded_at": "2026-04-11T15:00:00",
  "version": { ... }
}
```

---

## WebSocket: `ws://<server>:8000/ws/detect`

For real-time detection. Same logic as POST but over a persistent connection.

**Send:**
```json
{"text": "venmo me $20", "chat_id": "room_abc", "sender": "akash"}
```

**Receive:**
```json
{
  "text": "venmo me $20",
  "chat_id": "room_abc",
  "sender": "akash",
  "venmo_detection": {
    "is_money": true,
    "confidence": 0.9912,
    "trigger_type": "payment_app",
    "direction": "request",
    "detected_amount": "$20",
    "latency_ms": 189.4
  }
}
```

---

## Server Recommendations

| Scale | Instance | Specs | Cost | Latency |
|-------|----------|-------|------|---------|
| MVP (<500 users) | AWS `t3.medium` / GCP `e2-medium` | 2 vCPU, 4GB RAM | ~$30/mo | ~200-300ms |
| Growing (500-2000) | AWS `t3.large` / GCP `e2-standard-2` | 2 vCPU, 8GB RAM | ~$60/mo | ~150-250ms |
| Scale (2000+) | GPU instance or multiple CPU behind LB | varies | varies | ~20-50ms (GPU) |

CPU is fine for MVP. No GPU needed — model is only 67M parameters.

### Docker deployment

```bash
docker build -t paychat-model .
docker run -d -p 8000:8000 --name paychat paychat-model
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./saved_model` | Path to model files |
| `CONFIDENCE_THRESHOLD` | `0.65` | Min confidence to trigger (0.0-1.0) |

---

## Quick test

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "you owe me $20", "chat_id": "test123"}'
```
