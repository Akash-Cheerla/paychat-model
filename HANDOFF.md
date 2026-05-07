# Handoff — FYOE intent detection (v4)


## What it actually does

Takes a chat message, tells you which of **18 actionable intents** it hits, pulls out the **slots** (who, when, where, how much) from the text, and flags what's missing so the app can ask. One message can fire multiple intents at once — "remind me to venmo priya $25 at 8pm" fires both `money` and `alarm`.

Everything runs **on-device**. Backend never sees plaintext. E2EE stays intact.

Model is RoBERTa-base with 18 sigmoid heads. 19,001 training examples. Per-intent thresholds (not a flat 0.5 — each intent got its own optimal cutoff). 98.8% exact match on test set, 81.7% on a hand-crafted adversarial seed suite with 82 nasty edge cases. ~20-40ms per message on CPU.

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


## Slot filler

Every fired intent also gets **slots** pulled out — regex + dateparser, <5ms, zero cost. No LLM, no network call.

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
User sends message
    -> on-device model (18 sigmoid heads) -> fired intents
    -> on-device slot filler (regex + dateparser) -> extracted entities
    -> required slots filled? -> action card with deep link
    -> required slots missing? -> clarification card
    -> user taps -> deep link opens Venmo / Maps / Spotify / etc
```

Backend stores encrypted messages. Never decrypts, never runs the model.

**FYOE never touches money.** Deep-link to Venmo/PayPal.me/CashApp/UPI — payment happens in their app. No MSB licensing, no holding funds.


## Run the eval harness

```bash
pip install torch transformers fastapi uvicorn dateparser pydantic
python eval/eval_server.py
```

- `http://localhost:8001/` — type a message, see all 18 scores + slots
- `http://localhost:8001/chat` — two-person chat with live model annotations

Share with someone over the internet:
```bash
ngrok http 8001
# send the https://...ngrok-free.app/chat URL
```


## Files that matter

| file | what |
|---|---|
| `saved_model/` | v4 weights + thresholds (499MB, git LFS) |
| `eval/slot_filler.py` | slot extractor — pure regex + dateparser |
| `eval/eval_server.py` | FastAPI eval harness |
| `eval/test.html` | single-message tester UI |
| `eval/chat.html` | two-person chat UI |
| `training/generate_data.py` | data generator (19k examples) |
| `training/v4_failure_modes.py` | targeted failure-mode banks |
| `training/colab_train_v4.ipynb` | Colab notebook — T4 GPU, ~18 min |
| `v4_slot_filling_spec.md` | slot filling spec |


## Retraining

Open `training/colab_train_v4.ipynb` in Colab with a T4 runtime. Clones repo, generates data, trains 5 epochs (~18 min), runs seed suite, downloads zip. Drop the new `saved_model/` and you're done.


## When the model screws up

Send me the exact text + what you expected + what it actually did. I add it to the failure banks, regenerate, retrain. ~18 min cycle.
