# PayChat v21 — Mobile Integration

**For:** iOS & Android teams  
**What you receive:** Detection results attached to chat messages by the backend

---

## How It Works

```
User sends message → Backend calls /classify → Gets detection → Attaches to message payload → Your app receives enriched message
```

You never call the model API directly. The backend attaches detection data to each message.

---

## Message Payload (what you receive)

Your existing message object will have a new `detection` field:

```json
{
  "id": "msg_456",
  "text": "venmo me 30 bucks for dinner",
  "sender_id": 12,
  "room_id": "dm_12_45",
  "timestamp": "2026-07-04T10:30:00Z",
  "detection": {
    "intents": ["money"],
    "scores": {"money": 0.85},
    "slots": {"amount": "30 bucks", "note": "dinner"},
    "target": {"show_to": "others"},
    "money": {
      "detected_amount": "30 bucks",
      "trigger_type": "payment_app",
      "direction": "request"
    }
  }
}
```

If nothing detected: `detection.intents` is `[]` — no popup needed.

---

## When to Show a Popup

Two checks:

1. **Intent fired:** `detection.intents` is not empty
2. **Target matches current user:**

```swift
// iOS
let showTo = detection.target?.show_to ?? "sender"

switch showTo {
case "sender":
    // Show popup ONLY to the person who sent the message
    if currentUser.id == message.sender_id {
        showPopup(intent: detection.intents[0], slots: detection.slots)
    }
case "others":
    // Show popup to everyone EXCEPT the sender
    if currentUser.id != message.sender_id {
        showPopup(intent: detection.intents[0], slots: detection.slots)
    }
case "group":
    // Show popup to everyone
    showPopup(intent: detection.intents[0], slots: detection.slots)
default:
    break
}
```

```kotlin
// Android
val showTo = detection.target?.showTo ?: "sender"

when (showTo) {
    "sender" -> if (currentUser.id == message.senderId) showPopup(detection)
    "others" -> if (currentUser.id != message.senderId) showPopup(detection)
    "group" -> showPopup(detection)
}
```

---

## Intent → Popup Type

| Intent | Popup action | Example message |
|--------|-------------|-----------------|
| `money` | Open Venmo/payment flow | "venmo me 30 bucks" |
| `ride` | Open Uber/Lyft | "book me an uber to the airport" |
| `food_order` | Open DoorDash/UberEats | "order pizza from dominos" |
| `contact` | Open contacts/dialer | "call mom" |
| `alarm` | Open clock/alarm | "set alarm for 7am" |
| `reminder` | Create reminder | "remind me to submit report" |
| `calendar` | Open calendar | "block 2pm for meeting" |
| `bills` | Open bills/payment | "pay the electric bill" |
| `travel` | Open travel booking | "book a flight to NYC" |

---

## Slots — Pre-fill Popup Fields

Use `detection.slots` to pre-fill the action popup:

| Intent | Useful slots | Pre-fill |
|--------|-------------|----------|
| money | `amount`, `recipient`, `note` | Venmo amount + note |
| ride | `destination`, `pickup`, `time` | Uber destination |
| food_order | `food` (object) | Restaurant/item |
| alarm | `time` | Alarm time |
| reminder | `task`, `time` | Reminder text + time |
| calendar | `event`, `time` | Event name + time |
| bills | `bill_name`, `amount` | Bill type + amount |

---

## Money-Specific Fields

When `money` intent fires, you also get `detection.money`:

| Field | Values | Use |
|-------|--------|-----|
| `direction` | `"request"`, `"offer"`, `"split"` | Determines payment flow |
| `detected_amount` | `"$30"`, `"30 bucks"`, null | Pre-fill amount |
| `trigger_type` | `"payment_app"`, `"owing_debt"`, `"bill_splitting"`, `"direct_amount"`, `"general_money"` | Analytics |

**Direction logic:**
- `request` → sender is asking for money → recipient should pay
- `offer` → sender is offering to pay → sender initiates payment
- `split` → mutual split → everyone contributes

---

## Edge Cases

- **Multiple intents:** `intents` can have more than one. Show popup for the first/primary one, or let user pick.
- **No slots:** Sometimes intent fires but slots are null (e.g., "order food" without specifying what). Show a generic popup.
- **Guardrails:** If `detection.guardrails` is not null, a compliance issue was detected (credit card number in chat, etc.). You may want to show a warning instead of/in addition to the action popup.

---

## Detection Confidence

`detection.scores` has per-intent confidence (0.0–1.0). The model already applies thresholds, so anything in `intents` is above threshold. You don't need to filter by score — but you can use it for UI (e.g., dim the popup if score is barely above threshold).
