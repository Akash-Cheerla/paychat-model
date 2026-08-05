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

### ⚠️ What changes when the conversation classifier is enabled

The server can run the money/ride decision two ways. Which one is live is visible in
`conversation_state.decided_by`.

| | rule layer (today) | conversation classifier (`PAYCHAT_CONV_CLASSIFIER=1`) |
|---|---|---|
| `decided_by` | absent | `"conv_classifier"` |
| `status` values | `fired`, `pending`, `cancelled`, `reminder`, `no_fire` | **only `fired` and `no_fire`** |
| `triggered_by` | present on fire | present on fire — unchanged |
| group chats | need `reply_to`, else nothing fires | fire without `reply_to` |

**`target.show_to` is now always `"sender"` when `status == "fired"`**, with
`reason: "accepted_request"`. Under two-phase firing the message that fires IS the
acceptance, so the person who has to act is always the one who sent it. It previously
returned `"others"` on money — which pointed the payment sheet at whoever *asked* for
the money instead of the person who just agreed to send it. If you were already keying
off `message.senderId` for these (as the sample code below does) nothing changes for
you; if you were reading `target.show_to`, it is now correct.

**Nothing you already do breaks.** `status == "fired"` is still the trigger and
`triggered_by.slots` still carries the amount and destination. But three things go quiet:

* **`"pending"` never appears.** The classifier keeps no pending store — it re-reads the
  last 10 messages each time. A request simply produces `no_fire` until someone commits.
  Any UI that showed a "waiting for reply" state will never be entered.
* **`"cancelled"` never appears.** A rejection comes through as `no_fire`.
* **`"reminder"` never appears.** A deferral ("ill pay you friday") comes through as
  `no_fire`. If you show a reminder prompt today, it stops appearing.

**Group chats start firing.** Today nothing fires in a group unless the responder
swipe-replies to the request. The classifier does not need that. If prompts suddenly
appear in groups after this is switched on, that is the fix landing, not a bug.

Branch on `decided_by` if you need to support both, or gate the `pending`/`cancelled`/
`reminder` branches on its absence.

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
- **`conversation_state.triggered_by.slots`** — the values to actually use

```swift
// Prefer triggered_by slots — the acceptance ("sure") carries no amount of its own
let amount = convState.triggeredBy?.slots?["amount"]
    ?? detection.money?.detectedAmount

let note = convState.triggeredBy?.slots?["note"]
```

⚠️ **`triggered_by.slots` holds the EFFECTIVE values, not a transcript of the request.**
If the amount was negotiated after the request, this reflects what was agreed:

```
"can u lend me 2000"  ->  "i can only do 1000"  ->  "cool sending now"
    triggered_by.slots.amount == "$1000"      (agreed)
    triggered_by.text         == "can u lend me 2000"   (original wording)
```

The same applies to a ride destination that changed ("to koramangala" → "actually
make it indiranagar" → `destination: "Indiranagar"`), and to one that was never in
the request at all ("can you get me a cab" → "where to?" → "whitefield").

Read the slots, not the text. Parsing an amount or a place out of `triggered_by.text`
gives you the superseded value — on a payment sheet that means pre-filling double
what was agreed.

**Direction logic (for immediate money fires without state machine):**
- `request` → sender is asking for money → recipient should pay
- `offer` → sender is offering to pay → sender initiates payment
- `split` → mutual split → everyone contributes

---

## Group Chat vs DM

The state machine works differently in groups vs DMs. You don't need to handle this — the backend does. But know the behavior:

**DMs:** Response matching is automatic. Bob says "sure" after Alice requests → popup fires on Bob's screen. No special handling needed.

**Group chats — rule layer:** Responses only match when the user swipe-replies to the original request message. If Jake just types "bet" into a group with multiple pending requests, nothing fires — the system can't know which request he's responding to. When Jake swipe-replies to Maya's "venmo me 45" and says "bet", it fires correctly.

**Group chats — conversation classifier:** No `reply_to` needed. Maya asks the group, Jake replies "i got u, booking it now", and it fires with `target.show_to: "sender"` — the prompt goes to Jake, who is the one acting. Verified on a room id with no `dm_` prefix.

**Two requests open at once, answered with a bare "ok"** ("venmo me 20", then "book me a cab to hsr", then just "ok"): the classifier answers the **most recent** request — the ride, in that example. There is no explicit signal for which one is meant, and in real chat the last thing asked is usually the thing being answered. Naming the action — "ok sending" or "ok booking" — is unambiguous and always resolves to the right one.

The rule layer behaves differently here: it fires nothing. So this is one more case where `decided_by` changes what you see.

The backend passes `reply_to` (the replied-to message ID) to the model server. If your app already sends `reply_to_message_id` or similar in the message payload, make sure the backend is forwarding it. Otherwise money/ride in group chats won't fire.

**Next-day replies work too.** If someone replies to a money request the next day via swipe-reply, the server matches it from a 48-hour archive of expired requests.

---

## Edge Cases

- **Greeting after money request:** "Hi", "Hey", "Yo" etc. after a pending money request will NOT fire. The greeting guard catches these.
- **Multiple intents:** `intents` can have more than one. Show popup for the first/primary one.
- **No slots:** Sometimes intent fires but slots are null. Show a generic popup.
- **Guardrails:** If `detection.guardrails` is not null, a compliance issue was detected. You may want to show a warning.
- **`triggered_by` is null:** For self-initiated messages ("I'll venmo you") that fire without a prior pending request, `triggered_by` won't be present. Handle same as before.
- **Group chat with no `reply_to`:** Money/ride responses are silently ignored. No false fires — by design.

---

## Detection Confidence

`detection.scores` has per-intent confidence (0.0–1.0). The model already applies thresholds, so anything in `intents` is above threshold. You don't need to filter by score.
