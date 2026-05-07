# FYOE Integration Guide — v4


---

## How it fits together

Everything runs **on the user's device**.

```
Chat message typed
  -> on-device model (RoBERTa, 18 heads) -> which intents fired
  -> on-device slot filler (regex, <5ms) -> who, when, where, how much
  -> all required slots filled? -> show action card with deep link
  -> something missing? -> show clarification card ("who?", "how much?")
  -> user taps card -> deep link opens Venmo / Maps / Spotify / whatever
```

Backend just:
- Stores and relays encrypted messages (already doing this)
- Hosts model files for on-device download (~499MB, one-time)
- Doesn't decrypt, doesn't run the model, doesn't extract anything

---

## The 18 intents

| intent | fires when... | what the app does |
|---|---|---|
| `money` | debt, payment, splitting, venmo/cashapp mentions | deep-link Venmo / PayPal.me / CashApp / UPI |
| `alarm` | reminder, wake-up, timer | system alarm |
| `contact` | call, text, save number | phone / contacts intent |
| `calendar` | meeting, event with a time | calendar insert |
| `maps` | directions, place lookup | maps deep link |
| `ride` | cab, uber, lyft, ola | rideshare deep link |
| `food_order` | ordering food, swiggy, zomato | food delivery deep link |
| `travel` | flights, hotels, trips | booking app deep link |
| `shopping` | buying stuff online | Amazon / Flipkart deep link |
| `music` | play song/artist/playlist | Spotify / Apple Music deep link |
| `video` | watch show/movie/video | YouTube / Netflix deep link |
| `tickets` | event/movie tickets | BookMyShow / Fandango deep link |
| `reservation` | restaurant/hotel booking | OpenTable / Dineout deep link |
| `task` | to-do item | Todoist / Reminders |
| `note` | take a note | system notes |
| `bills` | utility/service bill | bill pay deep link |
| `health` | doctor, pharmacy, appointment | Practo / Zocdoc deep link |
| `weather` | weather check | weather card (no deep link) |

Multi-label — one message can fire 2+ intents at once.

---

## Model specs

- **RoBERTa-base** with 18 sigmoid heads
- 19,001 training examples with 8 failure-mode banks
- 98.8% IID exact match, 81.7% adversarial seed suite (82 edge cases)
- ~20-40ms on CPU, ~5-15ms on GPU
- ~499MB model weights

### Thresholds

**Don't use a flat 0.5.** Each intent has its own threshold from validation:

```json
{
  "money": 0.55, "alarm": 0.30, "contact": 0.20, "calendar": 0.70,
  "maps": 0.20, "food_order": 0.20, "ride": 0.50, "travel": 0.10,
  "shopping": 0.10, "music": 0.30, "video": 0.35, "tickets": 0.10,
  "reservation": 0.20, "task": 0.70, "note": 0.10, "bills": 0.20,
  "health": 0.45, "weather": 0.10
}
```

These are in `saved_model/thresholds.json`. If you hardcode 0.5, accuracy tanks.

---

## Slot filling

After the model fires intents, you extract entities from the text. This is pure regex + dateparser — no ML, no network, no cost.

### Example

`"venmo priya $25 for dinner"` -> fires `money`:
```json
{
  "amount": "25",
  "recipient": "priya",
  "note": "dinner",
  "method": "venmo",
  "direction": "send",
  "_required_filled": true
}
```

`"send money to someone"` -> fires `money`:
```json
{
  "amount": null,
  "recipient": null,
  "direction": "send",
  "_required_filled": false
}
// needs_clarification: [{ intent: "money", missing: "amount" }, { intent: "money", missing: "recipient" }]
```

### Slot schema (all 18)

| intent | required | optional |
|---|---|---|
| `money` | `amount`, `recipient` | `note`, `method`, `direction` |
| `alarm` | `datetime` | `note`, `recurrence` |
| `contact` | `name`, `channel` | `note` |
| `calendar` | `title`, `datetime` | `with_who`, `location` |
| `maps` | `destination` | `origin`, `mode` |
| `ride` | `destination` | `pickup`, `when`, `service` |
| `food_order` | `item` | `provider`, `when`, `address` |
| `travel` | `destination` | `origin`, `when`, `mode` |
| `shopping` | `item` | `qty`, `provider`, `when` |
| `music` | *(none)* | `song`, `artist`, `provider` |
| `video` | *(none)* | `title`, `provider` |
| `tickets` | `event` | `when`, `qty` |
| `reservation` | `place` | `when`, `qty` |
| `task` | `action` | `when` |
| `note` | `body` | |
| `bills` | `bill_kind` | `amount`, `due` |
| `health` | *(none)* | `provider`, `when`, `kind` |
| `weather` | *(none)* | `place`, `when` |

**If `_required_filled` is false, show clarification, don't guess.**

---

## Deep links — FYOE never touches money

We don't process, hold, or route payments. We just deep-link to the right app.

| detected method | deep link |
|---|---|
| venmo | `venmo://paycharge?txn=pay&recipients=<user>&amount=<amt>&note=<note>` |
| cashapp | `cashapp://cash.app/pay` |
| paypal | `https://paypal.me/<user>/<amt>` (use Custom Tab, not browser) |
| upi | `upi://pay?pa=<vpa>&am=<amt>&tn=<note>` |
| fallback | show picker based on user's default payment app |

For return-to-app: use `SFSafariViewController` (iOS) / Custom Tabs (Android) for web links. Register `fyoe://` callback scheme for native deep links.

Same pattern for non-payment intents — detect the provider from text ("on spotify", "from amazon")

---

## Android integration (the actual steps)

### 1. Get the model onto the device

~499MB. Either download on first launch (from your CDN) or bundle in APK. Files you need:
```
config.json, model.safetensors, tokenizer.json,
tokenizer_config.json, special_tokens_map.json, thresholds.json
```

### 2. Run inference

Use ONNX Runtime Mobile or PyTorch Mobile. Pseudocode:

```kotlin
val logits = model.forward(tokenize(text))  // float[18]
val fired = mutableListOf<String>()
for (i in LABELS.indices) {
    val prob = sigmoid(logits[i])
    if (prob >= thresholds[LABELS[i]]!!) {
        fired.add(LABELS[i])
    }
}
```

### 3. Port the slot filler

`eval/slot_filler.py` is ~400 lines of regex + dateparser. Port to Kotlin or run via Chaquopy if you want exact parity. No ML involved, just string matching.

### 4. Render cards

```kotlin
for (intent in fired) {
    val slots = extractSlots(text, intent)
    if (slots.requiredFilled) {
        showActionCard(intent, slots)  // tap -> deep link
    } else {
        showClarificationCard(intent, slots.missing)  // "who?", "when?"
    }
}
```

Multiple intents = multiple stacked cards, most confident first.

---


## Known limitations

1. **No conversation context** — model sees one message at a time. "How much?" after "I need to pay Priya" won't connect. Fix is on-device context window (planned).
2. **No coreference** — "Send it to her" doesn't know who "her" is. Same fix.
3. **81.7% adversarial** — that seed suite is intentionally brutal (sarcasm, negation, past tense). Real-world accuracy is higher.

---

