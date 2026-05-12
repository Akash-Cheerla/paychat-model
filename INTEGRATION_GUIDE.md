# FYOE Integration Guide — v5


---

## How it fits together

Everything runs on the **server (AWS Nitro Enclave)** — encrypted in, prediction out.

```
Chat message sent
  -> Server (Nitro Enclave) decrypts in secure enclave
  -> Trigger filter pre-screens (regex + fuzzy Levenshtein)
  -> Context accumulator prepends last 3-5 messages from conversation
  -> RoBERTa model (18 sigmoid heads) classifies with context
  -> Slot filler (regex + dateparser, <5ms) extracts entities
  -> All required slots filled? -> action card with deep link
  -> Something missing? -> clarification card ("who?", "how much?")
  -> Encrypted response sent back to client
  -> User taps card -> deep link opens Venmo / Maps / Spotify / etc
```

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

## Multi-turn context (v5)

The model now uses conversation history. This is the biggest upgrade — the model sees prior messages and uses them to classify the current message.

### What this enables

| Pattern | Before (v4) | After (v5) |
|---|---|---|
| "I'm hungry" → "dominos?" | Nothing fires | food_order fires |
| "order pizza?" → "yeah" | Nothing fires | food_order carries forward |
| "uber to JFK" → "make it lyft" | Confused | ride fires, app = lyft |
| "sure" after "order pizza?" | Nothing | food_order |
| "sure" after "good movie" | Nothing | Nothing (correct!) |

### How it works (server-side)

1. **Context accumulator** tracks last 5 messages per conversation (chat_id)
2. When a new message arrives, prior messages are prepended using `|` separator
3. The tokenizer encodes `(context, current_text)` as a text pair:
   ```
   <s> prior msg 1 | prior msg 2 </s></s> current message </s>
   ```
4. Model was trained on 2,486 examples in this exact format
5. Single-turn messages still work as before (no context = no pair encoding)

### Integration

```python
from context_accumulator import ConversationContext

ctx = ConversationContext(window_size=5)

# On each message:
context_str = ctx.get_context_string(chat_id)  # "msg1 | msg2" or None

if context_str:
    enc = tokenizer(context_str, text, max_length=128, ...)  # pair encoding
else:
    enc = tokenizer(text, max_length=128, ...)               # single

# After classification:
ctx.record(chat_id, text, fired=result["fired"], slots=result["slots"])
```

---

## Model specs

- **RoBERTa-base** with 18 sigmoid heads
- ~30,840 training examples (2,486 multi-turn context)
- 10 multi-turn conversation banks + 26 single-turn failure-mode banks
- Focal loss (gamma=2.0) + label smoothing (0.03) + mixed precision (fp16)
- 8 epochs on T4 GPU
- ~20-40ms on CPU, ~5-15ms on GPU
- ~499MB model weights

### Thresholds

**Don't use a flat 0.5.** Each intent has its own threshold from validation:

```json
{
  "money": 0.30, "alarm": 0.60, "contact": 0.45, "calendar": 0.29,
  "maps": 0.66, "food_order": 0.29, "ride": 0.57, "travel": 0.17,
  "shopping": 0.20, "music": 0.29, "video": 0.29, "tickets": 0.36,
  "reservation": 0.48, "task": 0.64, "note": 0.16, "bills": 0.64,
  "health": 0.20, "weather": 0.14
}
```

These are in `saved_model/thresholds.json`. If you hardcode 0.5, accuracy tanks.

---

## Trigger filter

`eval/trigger_filter.py` — pre-screens messages before they hit the model.

Two layers:
1. **Regex** — per-intent keyword patterns (fast, catches 95%+ of actionable messages)
2. **Fuzzy (Levenshtein distance 1)** — automatically catches typos/SMS-speak ("ordr" → "order", "ubr" → "uber", "piza" → "pizza")

Only keywords with 5+ characters get fuzzy-matched (avoids false positives like "good" → "food").

**With context:** When the context accumulator has prior messages for a conversation, the trigger filter is bypassed for that message — because context can make an otherwise-ambiguous message actionable ("dominos?" has no keywords but context makes it food_order).

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

Same pattern for non-payment intents — detect the provider from text ("on spotify", "from amazon").

---

## Server-side integration

### 1. Deploy model in Nitro Enclave

The model runs inside an AWS Nitro Enclave. Files needed:
```
config.json, model.safetensors, tokenizer.json,
tokenizer_config.json, special_tokens_map.json, thresholds.json
```

### 2. Run inference with context

```python
# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
thresholds = json.load(open(MODEL_DIR / "thresholds.json"))

# Per-message inference (with optional context)
context = "prior msg 1 | prior msg 2"  # from context accumulator, or None

if context:
    enc = tokenizer(context, text, return_tensors="pt", truncation=True, max_length=128)
else:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

logits = model(**enc).logits[0]
probs = torch.sigmoid(logits).tolist()
fired = [labels[i] for i, p in enumerate(probs) if p >= thresholds[labels[i]]]
```

### 3. Extract slots

```python
from slot_filler import extract_slots, needs_clarification

slots = extract_slots(text, fired)
missing = needs_clarification(slots)
```

### 4. Return result to client

```json
{
  "fired": ["food_order"],
  "slots": { "food_order": { "item": "pizza", "provider": "dominos" } },
  "needs_clarification": [],
  "context_used": true
}
```

---

## Known limitations (and what's next)

1. ~~**No conversation context**~~ **FIXED in v5** — model now sees last 3-5 messages
2. **No coreference** — "Send it to her" doesn't know who "her" is. Needs contact resolution against user's address book (backend Layer C).
3. **Context needs retraining** — v5 templates are the first version. Real user conversations will surface edge cases that need more training data.
4. **126 seed tests** — adversarial suite is intentionally brutal. Real-world accuracy is higher.

---
