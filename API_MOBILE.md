# PayChat v25 — Mobile Integration

**For:** iOS & Android teams  
**What you receive:** Detection results attached to chat messages by the backend

---

## How It Works

```
User sends message → Backend calls /classify → Gets detection → Attaches to message payload → Your app receives enriched message
```

You never call the model API directly. The backend attaches detection data to each message.

**Big change in v25:** Money and ride intents now fire on the *response*, not the request. If Alice says "venmo me $20", that message has no intent. When Bob replies "sure", *that* message carries the money intent + info about who requested it.

---

## Message Payload (what you receive)

Your existing message object will have a new `detection` field:

```json
{
  "id": "msg_456",
  "text": "sure sending now",
  "sender_id": 45,
  "room_id": "dm_12_45",
  "timestamp": "2026-07-20T10:30:00Z",
  "detection": {
    "intents": ["money"],
    "scores": {"money": 0.85},
    "slots": {},
    "target": {"show_to": "sender"},
    "money": null,
    "conversation_state": {
      "status": "fired",
      "response_type": "positive_ack",
      "triggered_by": {
        "sender": "12",
        "text": "venmo me 20 bucks for dinner",
        "message_id": "msg_440",
        "slots": {"amount": "20 bucks", "note": "dinner"}
      }
    }
  }
}
```

If nothing detected: `detection.intents` is `[]` — no popup needed.

---

## When to Show a Popup

Three checks now:

1. **Intent fired:** `detection.intents` is not empty
2. **Check conversation_state** if present — it tells you the full story
3. **Target matches current user**

### Handling `conversation_state`

```swift
// iOS
guard let intents = detection.intents, !intents.isEmpty else { return }

if let convState = detection.conversationState {
    switch convState.status {
    case "fired":
        // Someone acked a pending request — show popup to current sender
        let requester = convState.triggeredBy?.sender  // who asked for money
        let amount = convState.triggeredBy?.slots?["amount"]
        let note = convState.triggeredBy?.slots?["note"]
        
        if currentUser.id == message.senderId {
            // I'm the one who acked — show me the Venmo popup
            showPaymentPopup(
                payTo: requester,
                amount: amount,
                note: note
            )
        }
        
    case "reminder":
        // Someone said "I'll pay you friday" — set reminder
        if currentUser.id == message.senderId {
            let requester = convState.triggeredBy?.sender
            showReminderPopup(
                remindToPay: requester,
                amount: convState.triggeredBy?.slots?["amount"]
            )
        }
        
    case "pending":
        // Request stored, waiting for response — no popup yet
        break
        
    case "cancelled":
        // Rejection — clear any pending UI
        break
        
    default:
        break
    }
} else {
    // No conversation state — non-managed intent (food, alarm, etc.)
    // These fire immediately, use target.show_to like before
    handleImmediateIntent(detection)
}
```

```kotlin
// Android
val intents = detection.intents ?: return
if (intents.isEmpty()) return

val convState = detection.conversationState
if (convState != null) {
    when (convState.status) {
        "fired" -> {
            // Someone acked a pending request
            val requester = convState.triggeredBy?.sender
            val amount = convState.triggeredBy?.slots?.get("amount")
            val note = convState.triggeredBy?.slots?.get("note")
            
            if (currentUser.id == message.senderId) {
                showPaymentPopup(payTo = requester, amount = amount, note = note)
            }
        }
        "reminder" -> {
            if (currentUser.id == message.senderId) {
                val requester = convState.triggeredBy?.sender
                showReminderPopup(remindToPay = requester)
            }
        }
        "pending" -> { /* request stored, no popup */ }
        "cancelled" -> { /* rejection, clear pending UI */ }
    }
} else {
    // Non-managed intent — fire immediately
    handleImmediateIntent(detection)
}
```

### For non-managed intents (food, alarm, calendar, etc.)

These don't go through the state machine. No `conversation_state`. Use `target.show_to` like before:

```swift
func handleImmediateIntent(_ detection: Detection) {
    let showTo = detection.target?.showTo ?? "sender"
    
    switch showTo {
    case "sender":
        if currentUser.id == message.senderId {
            showPopup(intent: detection.intents[0], slots: detection.slots)
        }
    case "others":
        if currentUser.id != message.senderId {
            showPopup(intent: detection.intents[0], slots: detection.slots)
        }
    case "group":
        showPopup(intent: detection.intents[0], slots: detection.slots)
    default:
        break
    }
}
```

---

## Intent → Popup Type

| Intent | Popup action | State machine? | Example message |
|--------|-------------|----------------|-----------------|
| `money` | Open Venmo/payment flow | **Yes** — fires on response | "venmo me 30 bucks" |
| `ride` | Open Uber/Lyft | **Yes** — fires on response | "book me an uber to the airport" |
| `food_order` | Open DoorDash/UberEats | No — immediate | "order pizza from dominos" |
| `contact` | Open contacts/dialer | No — immediate | "call mom" |
| `alarm` | Open clock/alarm | No — immediate | "set alarm for 7am" |
| `reminder` | Create reminder | Special — also fires as response to future promises | "remind me to submit report" |
| `calendar` | Open calendar | No — immediate | "block 2pm for meeting" |
| `bills` | Open bills/payment | No — immediate | "pay the electric bill" |
| `travel` | Open travel booking | No — immediate | "book a flight to NYC" |

---

## Slots — Pre-fill Popup Fields

Use `detection.slots` for immediate intents. For money/ride fired from a response, use `conversation_state.triggered_by.slots` — those come from the original request which has the actual amount/destination.

| Intent | Useful slots | Pre-fill |
|--------|-------------|----------|
| money (from triggered_by) | `amount`, `recipient`, `note` | Venmo amount + note |
| ride (from triggered_by) | `destination`, `pickup`, `time` | Uber destination |
| food_order | `food` (object) | Restaurant/item |
| alarm | `time` | Alarm time |
| reminder | `task`, `time` | Reminder text + time |
| calendar | `event`, `time` | Event name + time |
| bills | `bill_name`, `amount` | Bill type + amount |

---

## Money-Specific Fields

When `money` intent fires on a response (status: "fired"), combine info from two places:

- **`detection.money`** — may be null on the response since "sure" isn't a money message itself
- **`conversation_state.triggered_by.slots`** — has the amount and note from the original request

```swift
// Get amount — prefer triggered_by slots since the original request has the real amount
let amount = convState.triggeredBy?.slots?["amount"]
    ?? detection.money?.detectedAmount

let note = convState.triggeredBy?.slots?["note"]
```

**Direction logic (for immediate money fires without state machine):**
- `request` → sender is asking for money → recipient should pay
- `offer` → sender is offering to pay → sender initiates payment
- `split` → mutual split → everyone contributes

---

## Edge Cases

- **Greeting after money request:** "Hi", "Hey", "Yo" etc. after a pending money request will NOT fire. The greeting guard catches these.
- **Multiple intents:** `intents` can have more than one. Show popup for the first/primary one.
- **No slots:** Sometimes intent fires but slots are null. Show a generic popup.
- **Guardrails:** If `detection.guardrails` is not null, a compliance issue was detected. You may want to show a warning.
- **`triggered_by` is null:** For self-initiated messages ("I'll venmo you") that fire without a prior pending request, `triggered_by` won't be present. Handle same as before.

---

## Detection Confidence

`detection.scores` has per-intent confidence (0.0–1.0). The model already applies thresholds, so anything in `intents` is above threshold. You don't need to filter by score.
