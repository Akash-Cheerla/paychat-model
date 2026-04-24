# PayChat — Intent Detection API (v2)

So basically I trained a DistilBERT model to pick up 5 actionable intents in chat messages — money, alarm, contact, calendar, and maps. One message can hit multiple intents at once ("remind me to venmo priya $25 at 8pm" fires both money *and* alarm). The API takes in a chat message and returns an array of detected intents with everything the Android app needs to fire the right system intent.

**Model info:**
- DistilBERT (distilbert-base-uncased), fine-tuned on ~4,900 examples
- 5 sigmoid heads, independent multi-label output
- ~20-40ms on CPU (same speed as the old single-head model — five heads share the same backbone)
- Confidence threshold at 0.5 per intent (you can change this)

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
python tests/test_cooldown.py
```
Runs 174 checks across 32 scenarios — cooldowns, per-intent isolation, multi-intent messages, websocket, error paths. Should all pass green. Takes about 2 seconds.

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
                { "text": "remind me to venmo priya $25 at 8pm",
                  "chat_id": "room123",
                  "sender": "akash" }
        |
        v
My API responds with detection data (intents[] array)
        |
        v
Backend attaches that to the message before sending to recipient:
{
  "id": "msg_456",
  "text": "remind me to venmo priya $25 at 8pm",
  "sender": "akash",
  "timestamp": "...",
  "intents": [
    { "type": "money", "should_popup": true, "payload": { "amount": "$25", ... }, "targeting": {...} },
    { "type": "alarm", "should_popup": true, "payload": { "time_iso": "...", "label": "venmo priya" }, "targeting": {...} }
  ]
}
        |
        v
Android iterates intents[] — for each where should_popup=true, fires the matching system intent
```

---

## POST /detect

This is the main one. Send every chat message here.

**Request:**
```json
{
  "text": "remind me to venmo priya $25 at 8pm",
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

Only `text` is required. `chat_id` is what keys the per-intent cooldown tracker — pass it consistently per conversation. `message_id` and `sender` are echoed back so you can match responses to messages.

**Response:**
```json
{
  "intents": [
    {
      "type": "money",
      "confidence": 0.98,
      "should_popup": true,
      "suppressed_reason": null,
      "cooldown_remaining_seconds": 0,
      "chat_state": "cooldown",
      "payload": { "amount": "$25", "trigger_type": "payment_app", "direction": "request" },
      "targeting": { "addressee": null, "third_party": "priya", "is_self": true, "is_mutual": false }
    },
    {
      "type": "alarm",
      "confidence": 0.97,
      "should_popup": true,
      "suppressed_reason": null,
      "cooldown_remaining_seconds": 0,
      "chat_state": "cooldown",
      "payload": { "label": "venmo priya", "time_iso": "2026-04-24T20:00", "time_phrase": "at 8pm" },
      "targeting": { "addressee": null, "third_party": "priya", "is_self": true, "is_mutual": false }
    }
  ],
  "is_money": true,
  "should_popup": true,
  "detected_amount": "$25",
  "trigger_type": "payment_app",
  "direction": "request",
  "confidence": 0.98,
  "chat_state": "cooldown",
  "latency_ms": 28.4,
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

**The v1 back-compat bit:** The flat top-level `is_money`, `should_popup`, `detected_amount`, etc. still exist — they mirror the money intent exactly like the old contract. If you already built for v1, nothing breaks. New code should read `intents[]`.

Errors: `400` if text is empty, `503` if the model is still loading on cold start.

---

## The 5 intents — what the app gets

Each `intents[]` entry has a `payload` shaped for its intent:

| intent | payload fields | android intent |
|---|---|---|
| `money` | `amount`, `trigger_type`, `direction` | venmo/cashapp/upi deep link |
| `alarm` | `label`, `time_iso`, `time_phrase`, `seconds_from_now?` | `AlarmClock.ACTION_SET_ALARM` |
| `contact` | `phone` (normalized), `name_hint` | `ContactsContract.Intents.Insert` |
| `calendar` | `title`, `start_iso`, `start_phrase`, `duration_minutes` | `CalendarContract.Events` insert |
| `maps` | `place` | `geo:` uri |

All of those are free Android system intents — no api keys, no external services. Just open the system app.

---

## The targeting object — who gets the popup

Every intent entry also has a `targeting` object:

```json
"targeting": { "addressee": "akash", "third_party": "priya", "is_self": false, "is_mutual": false }
```

- `addressee` — chat member the message is aimed at ("hey **akash** save this number")
- `third_party` — person mentioned who isn't a chat participant ("remind me to call **mom**")
- `is_self` — sender is referring to themselves ("remind **me**…")
- `is_mutual` — group action ("**team sync**", "**everyone** meet at…")

Use these to decide whose device actually pops:

```kotlin
// Android — picking who pops an alarm
if (intent.targeting.isSelf) {
    if (currentUserId == message.senderId) showAlarmPopup(intent.payload)
} else if (intent.targeting.isMutual) {
    showAlarmPopup(intent.payload)  // everyone
} else if (intent.targeting.addressee != null) {
    if (currentUsername == intent.targeting.addressee) showAlarmPopup(intent.payload)
}
```

For money specifically, the old `direction` field still works:

| direction | what's happening | who gets the popup |
|-----------|-----------------|-------------------|
| `request` | sender is asking for money ("you owe me $25") | everyone except sender |
| `offer` | sender wants to pay ("I'll send you $20") | the sender |
| `split` | splitting something ("let's split dinner") | everyone |

```kotlin
// Android — money direction handling (v1-compatible)
message.intents.firstOrNull { it.type == "money" }?.let { det ->
    if (!det.shouldPopup) return@let
    when (det.payload.direction) {
        "request" -> if (currentUserId != message.senderId) showVenmoPopup(det.payload)
        "offer"   -> if (currentUserId == message.senderId) showVenmoPopup(det.payload)
        "split"   -> showVenmoPopup(det.payload)
    }
}
```

---

## The cooldown — per chat, per intent

Server holds `(chat_id, intent)` state so related follow-up messages don't spam popups.

- First fire per (chat, intent) → pops
- Follow-ups within 5 min → suppressed (`cooldown_active`)
- A different payload for same intent → pops again (new action). dedupe key per intent: `amount`, `time_iso`, `phone`, `start_iso`, `place`
- User dismissed → 15 min cooldown
- Payment completes (money only) → cooldown clears, 60s grace ignores "sent!" / "thanks"

Critically — cooldowns are **independent per intent**. An alarm firing doesn't block a money popup in the same chat.

To make the cooldown feel smart, wire up these 3 endpoints from the backend:

- `POST /payment-complete/{chat_id}` — money only. Call on venmo/upi webhook or in-app payment success.
- `POST /popup-dismissed/{chat_id}?intent=<intent>` — user closed the popup. `intent` defaults to `money` for v1 callers.
- `POST /reset-cooldown/{chat_id}[?intent=<intent>]` — admin / testing / "snooze". Omit `intent` to clear all intents for the chat.

Skip these and the system still works — falls back to timer-only mode.

---

## Trigger types (money intent only)

| trigger_type | examples |
|-------------|----------|
| `owing_debt` | "you owe me $25", "pay me back" |
| `bill_splitting` | "let's split dinner", "halves?" |
| `direct_amount` | "that's $50", "30 bucks" |
| `payment_app` | "venmo me", "cashapp me" |
| `general_money` | "cover me", "spot me" |

---

## Other endpoints

**GET /health** — check if server is up and model loaded
```json
{
  "status": "healthy",
  "num_labels": 5,
  "intents": ["money", "alarm", "contact", "calendar", "maps"],
  "version": { "trained_at": "2026-04-24", "test_accuracy": 0.98, "test_f1": 0.97 },
  "threshold": 0.5
}
```

**GET /metrics** — request counts, per-intent detection counts, popup stats
```json
{
  "requests": 142,
  "money_detected": 37,
  "intents_detected": { "money": 37, "alarm": 12, "contact": 5, "calendar": 22, "maps": 18 },
  "popups_fired": 48,
  "popups_suppressed": 46,
  "suppression_rate": 0.489,
  "avg_latency_ms": 29.1
}
```

**GET /chat-state/{chat_id}** — per-intent cooldown state for a chat. Useful when debugging "why didn't that popup fire?".

**POST /reload** — if we retrain the model, call this to swap it in without restarting the server.

**WebSocket /ws/detect** — same as POST /detect but over websocket. Send JSON, get JSON back with `intents[]` + v1-mirror `venmo_detection` object attached.

---

## Deployment notes

**Env vars:**
- `MODEL_DIR` — path to the saved_model folder (defaults to `./saved_model`)
- `CONFIDENCE_THRESHOLD` — defaults to 0.5 per intent (was 0.65 in v1 for single-class softmax; sigmoids need a different baseline)
- `POPUP_COOLDOWN_SECONDS` — 300 default (quiet window per intent)
- `DISMISSED_COOLDOWN_SECONDS` — 900 default
- `POST_PAYMENT_GRACE_SECONDS` — 60 default
- `TRACKER_EVICTION_SECONDS` — 1800 default (drops idle chat trackers)

**Model files** are in `saved_model/`:
```
config.json, model.safetensors (255 MB), tokenizer.json, tokenizer_config.json, training_report.json
```

Weights file is big (255 MB) so if you're using git, you'll need LFS:
```bash
git lfs install
git lfs track "saved_model/model.safetensors"
```

**For prod** — GPU instance (like AWS g4dn.xlarge) does inference in 5-15ms instead of 20-40ms on CPU. Use `--workers 1` since the model loads per worker.

Hit `/health` and confirm `num_labels: 5` before sending traffic. The server also handles a legacy 2-class (money-only) checkpoint — `num_labels` will read `2` and only the money head fires.

**Docker compose:**
```yaml
services:
  paychat-detector:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CONFIDENCE_THRESHOLD=0.5
    restart: unless-stopped
```

---

## Retraining

If we ever add more intents or need to retrain:
```bash
cd training/
pip install -r requirements.txt
python generate_data.py    # regenerates training data for all 5 intents
python train.py            # trains a multi-label (5-head) model
```
Then copy the new `saved_model/` over and hit `POST /reload`. Zero-downtime hot-swap.

---

## What each intent catches vs ignores

- **money** — picks up debt/owing, bill splitting, amounts in payment context, payment-app mentions (venmo/cashapp/zelle/upi), general asks (cover me, spot me). Ignores: stock talk, prices without payment intent ("that shirt is $50"), random numbers ("gate 12").
- **alarm** — picks up "remind me", "wake me up", "set alarm", "ping me in 20 min", "don't let me forget". Ignores: past references ("reminded me of X"), generic "hey" messages.
- **contact** — picks up "save this number" + a phone number pattern. US (`+1 415-555-1234`, `(415) 555-1234`) and India (`+91 98765 43210`). Ignores: credit card digits, order numbers, ages.
- **calendar** — picks up "meeting at 3pm tomorrow", "team sync friday 4pm", "dinner sat 8pm", "wedding on april 30". Ignores: vague "let's hang out" without a time.
- **maps** — picks up "meet me at X", "heading to X", "directions to X", "i'm at X", addresses. Ignores: metaphorical uses ("meet you halfway").

---

## Swagger docs

Once the server is running: `http://<server>:8000/docs`

Hit me up if anything's unclear — Akash
