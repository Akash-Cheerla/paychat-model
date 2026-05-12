# Handoff — FYOE intent detection (v5)


## What it actually does

Takes a chat message **plus conversation context**, tells you which of **18 actionable intents** it hits, pulls out the **slots** (who, when, where, how much) from the text, and flags what's missing so the app can ask. One message can fire multiple intents at once — "remind me to venmo priya $25 at 8pm" fires both `money` and `alarm`.

**v5's big addition:** The model now understands conversation context. "dominos?" alone = nothing, but after "bro I'm hungry" + "same" it fires `food_order`. Confirmations ("yeah do it" after "should we order pizza?") carry forward the prior intent. The model was trained on 2,486 multi-turn examples across 10 conversation pattern banks.

Everything runs on the **server (AWS Nitro Enclave)**. Encrypted in, prediction out. Even we can't see user messages during inference.

Model is RoBERTa-base with 18 sigmoid heads. ~30,840 training examples (including 2,486 multi-turn context examples). Per-intent thresholds (not a flat 0.5 — each intent got its own optimal cutoff). 8 epochs with focal loss + label smoothing + mixed precision. ~20-40ms per message on CPU.

The 18 intents:

| intent | example | what the app does |
|---|---|---|
| `money` | "you owe me $20", "venmo me" | deep-link to Venmo / PayPal.me / CashApp / UPI |
| `alarm` | "remind me at 10pm" | set system alarm |
| `contact` | "call dad", "save this number" | call / text / save contact |
| `calendar` | "meeting at 3pm tomorrow" | add calendar event |
| `maps` | "heading to SFO" | open in maps |
| `ride` | "book an uber to airport" | Uber / Lyft / Ola deep link |
| `food_order` | "order pizza from dominos" | Swiggy / Zomato / DoorDash deep link |
| `travel` | "flights to goa next friday" | booking app deep link |
| `shopping` | "order 2 hdmi cables" | Amazon / Flipkart deep link |
| `music` | "play anti-hero by taylor swift" | Spotify / Apple Music deep link |
| `video` | "put on the office on netflix" | YouTube / Netflix deep link |
| `tickets` | "2 tickets for oppenheimer" | BookMyShow / Fandango deep link |
| `reservation` | "table for 4 at olive garden" | OpenTable / Dineout deep link |
| `task` | "add buy groceries to my todo" | create task in Todoist / Reminders |
| `note` | "note: meeting moved to 3pm" | create note |
| `bills` | "pay electricity bill" | bill pay deep link |
| `health` | "book a dentist appointment" | Practo / Zocdoc deep link |
| `weather` | "will it rain in mumbai tomorrow" | show weather card |


## Multi-turn context (v5)

The model now classifies messages using conversation history, not just the current message. This is the biggest upgrade since v3.

### How it works

Every message gets the last 3-5 prior messages prepended as context using the tokenizer's text-pair encoding:

```
<s> prior msg 1 | prior msg 2 </s></s> current message </s>
```

The model was trained on this format with 2,486 multi-turn examples across 10 conversation patterns:

| Pattern | Example | What happens |
|---|---|---|
| **Context buildup** | "I'm hungry" → "same" → "dominos?" | food_order fires (bare "dominos?" alone = nothing) |
| **Confirmation** | "order pizza?" → "yeah do it" | food_order carries forward |
| **Correction** | "uber to JFK" → "actually make it lyft" | ride stays, app changes to lyft |
| **Disambiguation** | "sure" after "order pizza?" = food_order. "sure" after "good movie" = nothing | Same word, different outcome based on context |
| **No carryforward** | "order pizza" → "nice game last night" | Nothing fires (casual after actionable) |

### Context accumulator

`eval/context_accumulator.py` manages per-conversation history. It tracks messages, their fired intents, and slots. The eval server WebSocket chat uses it automatically — every message gets context from prior messages in the same room.

For the production API (`app.py`), the existing `chat_history` dict and `_build_context_text()` function need to be updated to use the tokenizer's pair encoding instead of naive `" | "` concatenation. The `context_accumulator.py` module can be imported directly.


## Slot filler

Every fired intent also gets **slots** pulled out -- regex + dateparser, <5ms, zero cost. No LLM, no network call.

`"venmo priya $25 for dinner"` fires `money` with:
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

If required slots are missing, app shows a clarification card ("who?", "how much?") instead of guessing. The `_required_filled` flag tells you which case you're in.


## Architecture

```
User sends message in chat
  -> Server (Nitro Enclave) receives encrypted message
  -> Trigger filter (regex + fuzzy Levenshtein) pre-screens
  -> Context accumulator prepends last 3-5 messages
  -> RoBERTa model (18 sigmoid heads) classifies with context
  -> Slot filler (regex + dateparser) extracts entities
  -> Required slots filled? -> action card with deep link
  -> Required slots missing? -> clarification card
  -> Encrypted response sent back to client
  -> User taps card -> deep link opens Venmo / Maps / Spotify / etc
```

The Nitro Enclave ensures even we can't see plaintext during inference. Encrypted in, prediction out.


## Trigger filter

`eval/trigger_filter.py` — a two-layer keyword gatekeeper that pre-screens messages before hitting the model:

1. **Layer 1: Regex** — fast keyword matching across all 18 intents
2. **Layer 2: Fuzzy (Levenshtein distance 1)** — catches typos and SMS-speak automatically ("ordr" → "order", "ubr" → "uber")

If a message has no action-related keywords AND no context (when running with context accumulator), it's skipped entirely. This filters out ~80% of casual chat.


## Files that matter

| file | what |
|---|---|
| `saved_model/` | v5 weights + thresholds (~499MB) |
| `eval/context_accumulator.py` | **NEW** — multi-turn conversation context tracker |
| `eval/slot_filler.py` | slot extractor — pure regex + dateparser |
| `eval/trigger_filter.py` | keyword pre-screen + fuzzy matching |
| `eval/eval_server.py` | FastAPI eval harness with context-aware WebSocket chat |
| `eval/test.html` | single-message tester UI |
| `eval/chat.html` | two-person chat UI (uses context accumulator) |
| `eval/seed_tests.jsonl` | 126 adversarial seed tests (16 multi-turn) |
| `training/generate_data.py` | data generator (~30.8K examples) |
| `training/v4_failure_modes.py` | targeted failure-mode banks (v4.0-v4.3) |
| `training/v5_multiturn.py` | **NEW** — 241 multi-turn conversation templates |
| `training/train.py` | trainer with multi-turn support, focal loss, fp16 |
| `training/colab_train_v4.ipynb` | Colab notebook — T4 GPU, ~25-30 min |
| `app.py` | production API (needs context accumulator update) |


## Retraining

Open `training/colab_train_v4.ipynb` in Colab with a T4 runtime. Clones repo (v5 branch), generates data (~30.8K examples including 2,486 multi-turn), trains 8 epochs with focal loss + fp16, runs seed suite, downloads zip. Drop the new `saved_model/` and you're done.


## Version history

| version | date | what changed |
|---|---|---|
| v3 | Apr 2026 | 18 intents, RoBERTa-base, 99.06% IID, 52.4% seed suite |
| v4.0 | Apr 2026 | Failure-mode banks (negation, past tense, multi-intent, code-mixed), 81.7% seed |
| v4.1 | Apr 2026 | Teammate feedback fixes, SMS-speak, word amounts |
| v4.2 | May 2026 | Contrastive training, 15 targeted failure fixes |
| v4.3 | May 2026 | Focal loss, question forms, idiomatic negatives, 98.9% IID, ~98% seed |
| **v5** | **May 2026** | **Multi-turn context (241 templates, 10 banks), 30.8K examples, 126 seed tests** |


## When the model screws up

Send the exact text + context (prior messages) + what you expected + what it actually did. I add it to the failure banks, regenerate, retrain.
