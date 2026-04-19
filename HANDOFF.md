# Handoff — PayChat money detection


## What it actually does fyi

Takes a chat message, tells you if it's about money, and tells you whether to actually pop the venmo sheet. That last part matters — i'll explain below.

Model is DistilBERT fine-tuned on ~5400 chat examples. Currently 99.81% accuracy on the test set, 100% precision. Good enough to ship.

## Run it

```
docker build -t paychat-model .
docker run -d -p 8000:8000 --name paychat paychat-model
```

First boot takes ~20s to load the model into memory. After that every request is ~200-300ms on CPU. We're fine on a t3.medium for MVP, don't need a GPU.

`GET /health` returns 503 while it's loading and 200 once the model is ready. Use that for your readiness probe.

## The only field you need to read

Every chat message goes through:

```
POST /detect
{ "text": "you owe me $20", "chat_id": "room_xyz", "sender": "akash" }
```

Response has a bunch of fields but **the only one you need to check is `should_popup`**. True = pop venmo. False = don't. That's it.

```json
{
  "is_money": true,
  "should_popup": true,
  "detected_amount": "$20",
  "direction": "request",
  "trigger_type": "owing_debt",
  ...
}
```

The rest is diagnostic — good for logs, not required for branching.

## Why there's a `should_popup` separate from `is_money`

Coz people send 10 money messages in a row sometimes. "you owe me 20" → "bro pay up" → "venmo me fr" → "lol send it". If we popped venmo for every single one the user would lose their mind. So the server holds a per-chat state machine and only says `should_popup: true` when it actually makes sense.

You don't need to know the internal rules to integrate. But in case anyone asks:

- First money msg in a chat → pops
- Follow-ups inside a 5 min window → suppressed (quietly, no user-visible anything)
- A *different* $ amount in the same chat → pops again (new transaction)
- Payment completes → cooldown clears
- User dismissed without paying → 15 min cooldown before we try again

Net result in a messy conversation: 1-2 popups instead of 10. Which is the whole point.

## 3 endpoints you need to call besides /detect

The server is smart about timing but only if you tell it when real events happen. Wire these up:

**Payment succeeded** — when venmo/cashapp webhook confirms, or your in-app payment flow hits the success screen:
```
POST /payment-complete/{chat_id}
{ "amount": "$40", "payer": "rohit", "method": "venmo" }
```
Body is optional, fill it if you want audit info in logs.

**User dismissed the popup without paying** — they tapped X, closed the sheet, etc:
```
POST /popup-dismissed/{chat_id}
```
No body.

**Force-clear cooldown** — for testing or admin flows:
```
POST /reset-cooldown/{chat_id}
```

If you skip #1 and #2, the system still works, it just falls back to dumb timer-only mode. Wiring them up is what makes it feel smart.

## Debug endpoint

If you're ever like "why didn't that popup fire" or "why did it fire twice":

```
GET /chat-state/{chat_id}
```

Tells you exactly what the server thinks about that chat — current state, cooldown remaining, last amount seen, total popups fired, total suppressed. Check this before pinging me.

## Websocket (same thing, realtime)

```
ws://host:8000/ws/detect
```

Send `{"text": "...", "chat_id": "...", "sender": "..."}`, receive the same fields wrapped inside a `venmo_detection` object. Same popup rules apply. Use http or ws, pick whichever fits the app better, the contract is identical.

## Env vars worth knowing

| var | default | what it does |
|---|---|---|
| `POPUP_COOLDOWN_SECONDS` | 300 | quiet window after a popup fires |
| `DISMISSED_COOLDOWN_SECONDS` | 900 | longer window if user dismissed |
| `POST_PAYMENT_GRACE_SECONDS` | 60 | brief suppression after `/payment-complete` so "sent!" / "thanks" don't re-pop |
| `CONFIDENCE_THRESHOLD` | 0.65 | min model confidence to call it money |
| `MODEL_DIR` | `./saved_model` | where the weights live |

Tune if you want, but the defaults are what we tested against.

## Scale note

The cooldown tracker is in-memory. Single-instance deploy is fine for MVP and keeps everything fast — no redis hop. When we scale past 1 server (load balancer, multiple replicas), the tracker needs to move to redis so all instances share state. Swap the `popup_tracker` dict in `app.py` for a redis client with the same keys. Nothing else changes. Not a big job, just not MVP.

## Metrics

```
GET /metrics
```

Gives you requests count, money detection rate, popups fired, popups suppressed, suppression rate, active chat count. Hook it into whatever dashboard.

If `suppression_rate` is ~0 in production, that's a red flag — probably means `chat_id` isn't being passed consistently, so the tracker can't dedupe. Check that first.

## When the model is wrong

Two failure modes:

1. **False positive** — says is_money when it isn't. Low priority, just a stray popup.
2. **False negative** — misses a real money message. Higher priority, user doesn't get the popup they expect.

Send me the exact text that failed + what you expected. I'll add it to training data and retrain — takes ~10 min on colab gpu. Then i push the new `saved_model/` folder to the repo and you hit `POST /reload` to hot-swap without restarting the server. Zero downtime.

## Running the tests

```
python tests/test_cooldown.py
```

98 checks across 20 scenarios covering the full popup lifecycle — cooldown, dismissal, payment-complete, grace window, multi-chat isolation, websocket parity, everything. Takes about 1 second.

If you change anything in `app.py`, run this first. If any check fails, something's broken.

## Files that matter

- `app.py` — the entire api, everything lives here
- `saved_model/` — the weights (255mb, tracked with git lfs)
- `API_DOCS.md` — field-by-field reference if you want the details
- `tests/test_cooldown.py` — the test suite
- `Dockerfile` — deploys as-is

Everything else (`training/`, the notebook, dataset json files) is training-side stuff. You don't need any of it to run the api.

## Quick integration checklist

- [ ] Docker built, running, `/health` returns 200
- [ ] App sends every chat message to `POST /detect` with `chat_id`
- [ ] App reads `should_popup` to decide whether to show the venmo sheet
- [ ] App hits `/payment-complete/{chat_id}` when a payment succeeds
- [ ] App hits `/popup-dismissed/{chat_id}` when user closes the popup
- [ ] Load balancer health check points at `/health`
- [ ] `/metrics` wired into the dashboard

That's it. If all 7 are ticked we're good to demo.

## please test this before demo

- Run the test suite once to confirm nothing regressed
- Hit `/health` to confirm the model's loaded and version is the latest
- Open a real chat, send "you owe me $20", watch the popup fire
- Send 3 more nags in the same chat, confirm nothing fires
- Pay it, send "also $15 for gas", confirm the new popup fires

Basically live the example convo. If all 5 look right, we're shipping.
