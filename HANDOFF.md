# Handoff — PayChat intent detection (v2)


## What it actually does fyi

Takes a chat message, tells you which of 5 actionable intents it hits, and tells you whether to actually pop the sheet for each. That last part matters — i'll explain below.

The 5 intents and what each one triggers on Android:

| intent | example message | android action |
|---|---|---|
| `money` | "you owe me $20", "venmo me" | venmo / cashapp / upi flow |
| `alarm` | "remind me at 10pm" | `AlarmClock.ACTION_SET_ALARM` |
| `contact` | "save this number +1 415-555-1234" | `ContactsContract.Intents.Insert` |
| `calendar` | "meeting at 3pm tomorrow" | `CalendarContract.Events` insert |
| `maps` | "heading to dolores park" | `geo:` uri / google maps |

All 5 are free android intents — no api keys, no external services, just open the system app. Model is a DistilBERT with 5 sigmoid heads (multi-label so one message can fire money + alarm at once, like "remind me to venmo priya at 8pm $25"). ~4900 training examples. Not retrained yet but the old 2-class money checkpoint works fine with the new api too — num_labels auto-detect.

## Run it

```
docker build -t paychat-model .
docker run -d -p 8000:8000 --name paychat paychat-model
```

First boot takes ~20s to load the model. After that every request is ~20-40ms on cpu (five heads, same backbone, no slower than the single-head version). t3.medium is fine for mvp.

`GET /health` returns 503 while loading, 200 when ready. Use for readiness probe. Response now includes `num_labels` and `intents` array so you can see what the server thinks it's serving.

## The one field you read per intent

Every chat message goes through:

```
POST /detect
{ "text": "remind me to venmo priya $25 at 8pm", "chat_id": "room_xyz", "sender": "akash" }
```

Response has an `intents` array with one entry per intent that fired. Each entry has `should_popup`. **That's the only field the app branches on.** True = fire the popup for that intent. False = stay quiet.

```json
{
  "intents": [
    { "type": "money",  "should_popup": true, "payload": { "amount": "$25", ... }, "targeting": {...} },
    { "type": "alarm",  "should_popup": true, "payload": { "time_iso": "2026-04-24T20:00", "label": "venmo priya" }, "targeting": {...} }
  ],
  "is_money": true, "should_popup": true, "detected_amount": "$25", ...  // v1 back-compat mirror of money
}
```

Everything else (`confidence`, `trigger_type`, `cooldown_remaining_seconds`, etc.) is diagnostic — logs, dashboards, debugging — not required for branching.

**Existing v1 code keeps working.** The flat top-level fields (`is_money`, `should_popup`, `detected_amount`, etc.) still mirror the money intent exactly like before. You can migrate to the `intents[]` array when you have time. Nothing breaks in the meantime.

## Per-intent payloads — what the app gets

Each `intents[].payload` is pre-built so android can fire the system intent with no extra parsing:

- **money** → `{ amount, trigger_type, direction }` — pass `amount` to the venmo/upi deep link
- **alarm** → `{ label, time_iso, time_phrase, seconds_from_now }` — `time_iso` → hour/min for `AlarmClock.EXTRA_HOUR` / `EXTRA_MINUTES`. `label` → `EXTRA_MESSAGE`
- **contact** → `{ phone, name_hint }` — `phone` is already normalized to `+1 415 555 1234` or `+91 98765 43210`. pass to `ContactsContract.Intents.Insert.PHONE`
- **calendar** → `{ title, start_iso, start_phrase, duration_minutes }` — `start_iso` → `CalendarContract.EXTRA_EVENT_BEGIN_TIME`. duration is 30 for meetings, 60 for dinners
- **maps** → `{ place }` — url-encode and pass to `geo:0,0?q=<place>`. let google maps resolve it

Full Android snippets for all 5 are in `API_DOCS.md`.

## Targeting — who gets the popup

Every intent also comes with a `targeting` object:

```json
"targeting": { "addressee": "akash", "third_party": null, "is_self": false, "is_mutual": false }
```

- `addressee` — the chat member the message is aimed at ("hey **akash** save this number")
- `third_party` — a person mentioned who isn't in the chat ("call **mom**")
- `is_self` — sender is talking about themselves ("remind **me**…")
- `is_mutual` — group action, pop for everyone ("**team sync** friday 4pm", "**everyone** meet at the mission")

Use these to decide whose device actually pops. A "remind me" alarm should only fire on the sender's phone; a "team sync" calendar event should fire for everyone in the chat.

## Why `should_popup` exists separate from detection

Coz people send 10 related messages in a row. "you owe me 20" → "bro pay up" → "venmo me fr" → "lol send it". The model says yes to all of them. If we popped for every single one the user would lose their mind. So the server holds **per-(chat, intent) state** and only says `should_popup: true` when it's useful.

Rules per intent (they're independent — alarm cooldown never blocks a money popup in the same chat):

- First intent-fire in a chat → pops
- Follow-ups inside a 5 min window → suppressed quietly
- A *different* payload for the same intent → pops again (new transaction). dedupe key is per-intent:
  - money → `amount`
  - alarm → `time_iso`
  - contact → `phone`
  - calendar → `start_iso`
  - maps → `place`
- user dismissed without acting → 15 min cooldown
- money payment completes → cooldown clears, 60s grace so "sent!" / "thanks" don't re-pop

Net result: 1-2 popups per intent per chat, instead of 10. Which is the whole point.

## 3 endpoints you call besides /detect

**Payment succeeded** — money intent only. When venmo/cashapp/upi confirms:
```
POST /payment-complete/{chat_id}
{ "amount": "$40", "payer": "rohit", "method": "venmo" }
```
Body optional. Only affects the money cooldown — alarm/calendar/etc in the same chat are untouched.

**User dismissed a popup** — now scoped to the specific intent:
```
POST /popup-dismissed/{chat_id}?intent=alarm
```
If you omit `?intent=`, defaults to `money` for v1 back-compat.

**Force-clear cooldown** (testing/admin):
```
POST /reset-cooldown/{chat_id}?intent=money    # just money
POST /reset-cooldown/{chat_id}                  # all intents for the chat
```

Skip these and the system falls back to dumb-timer-only. Wiring them up is what makes it feel smart.

## Debug endpoint

If you're ever "why didn't that popup fire" or "why twice":

```
GET /chat-state/{chat_id}
```

Returns per-intent state — which intents are in cooldown, which are dismissed, last payload for each, time remaining. Before pinging me, check this.

## Websocket (same thing, realtime)

```
ws://host:8000/ws/detect
```

Send `{"text": "...", "chat_id": "...", "sender": "..."}`, receive the full response with `intents[]` + v1 `venmo_detection` mirror. Same cooldown rules. Http or ws, pick whichever fits the app better — contract is identical.

## Env vars worth knowing

| var | default | what it does |
|---|---|---|
| `POPUP_COOLDOWN_SECONDS` | 300 | quiet window after any popup fires |
| `DISMISSED_COOLDOWN_SECONDS` | 900 | longer window if user dismissed |
| `POST_PAYMENT_GRACE_SECONDS` | 60 | brief suppression after `/payment-complete` so "sent!" / "thanks" don't re-pop |
| `CONFIDENCE_THRESHOLD` | 0.5 | per-intent sigmoid threshold (independent heads) |
| `MODEL_DIR` | `./saved_model` | where the weights live |

Tune if you want, defaults are what we tested against.

## Scale note

Cooldown tracker is in-memory, keyed by `(chat_id, intent)` tuples. Single-instance is fine for mvp, keeps things fast, no redis dep. When we scale past 1 server, swap `popup_tracker` in `app.py` for redis with keys like `paychat:{chat_id}:{intent}`. Nothing else changes. Not a big job, just not mvp.

## Metrics

```
GET /metrics
```

Now includes `intents_detected` dict (one counter per intent) on top of the v1 money counters. Hook into whatever dashboard.

If `suppression_rate` is ~0 in prod — probably means `chat_id` isn't being passed consistently so the tracker can't dedupe. Check that first.

## When the model is wrong

Same two failure modes per intent:

1. **False positive** — fires an intent it shouldn't. Low priority, just a stray popup the user dismisses.
2. **False negative** — misses a real intent the user expected. Higher priority.

Send me the exact text + what intent you expected + what actually fired. I'll add to training data and retrain — ~15 min on colab gpu for the 5-head model. Push the new `saved_model/`, you hit `POST /reload`, zero downtime.

## Running the tests

```
python tests/test_cooldown.py
```

**174 checks across 32 scenarios** — money back-compat (20), per-intent cooldown isolation, multi-intent messages, scoped `/popup-dismissed` and `/reset-cooldown`, targeting extraction, intents[] shape, websocket parity, error paths. Takes ~2 seconds. Mocks the model so no gpu / no model load needed.

If you change `app.py`, run this first. If anything fails, something broke.

## Files that matter

- `app.py` — entire api, everything lives here
- `saved_model/` — weights (255mb, git lfs)
- `API_DOCS.md` — field-by-field reference + android code snippets per intent
- `tests/test_cooldown.py` — the test suite
- `training/` — dataset generator + training script for when we add more intents
- `Dockerfile` — deploys as-is

## Quick integration checklist

- [ ] Docker built, running, `/health` returns 200 with `num_labels: 5`
- [ ] App sends every chat message to `POST /detect` with `chat_id` + `sender`
- [ ] App iterates `intents[]` and calls the relevant android intent for each entry where `should_popup: true`
- [ ] App hits `/payment-complete/{chat_id}` when a payment succeeds
- [ ] App hits `/popup-dismissed/{chat_id}?intent=<intent>` when a user closes a popup
- [ ] App uses `targeting.is_self` / `is_mutual` to decide whose device fires the popup for alarms/maps
- [ ] Load balancer health check points at `/health`
- [ ] `/metrics` wired into the dashboard

That's it. If all 8 are ticked we're good to demo.

## please test this before demo

- Run the test suite once, confirm 174/174
- Hit `/health`, confirm model loaded + `num_labels: 5`
- Open a real chat, send each of these and watch the right popup fire:
  - `"you owe me $20"` → venmo popup
  - `"remind me to take meds at 10pm"` → alarm popup
  - `"save this number +1 415-555-1234"` → contact popup
  - `"meeting at 3pm tomorrow"` → calendar popup
  - `"heading to dolores park"` → maps popup
- Send a multi-intent msg: `"remind me to venmo priya $25 at 8pm"` → both money and alarm popups fire
- Send 3 follow-up nags in the same chat, confirm nothing fires
- Pay the $20, send "also $15 for gas", confirm the new money popup fires

Basically live the demo convo. If all of that looks right, we're shipping.
