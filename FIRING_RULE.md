# Firing rule — the single source of truth

Confirmed with the product owner on 2026-07-31. **Revised 2026-08-02** — see §3a, §3b
and §6a. Every generated label, every eval label, and every training target derives from
**this document**. If a labelling script disagrees with this file, the script is wrong.

Scope: **money** and **ride** only.

## The 2026-08-02 revision, in one line

> **Fire only when an action is still needed.** Reporting something already done is
> information, not a trigger.

This reverses two earlier rows. `just sent it on gpay` used to fire and no longer does
(§3a); an ambiguous `ok` with two open requests used to fire nothing and now resolves to
the most recent request (§6a).

**The first reversal is not a relabelling job.** The old rule shipped into all 117,677
training windows, so the model scores `just sent` at 0.996 — it is obeying the spec it
was trained on. Correcting it requires regenerating the training data and retraining;
fixing eval labels alone will not move the model.

---

## The one principle

> **Fire only when there is an actionable cause, and the responsible party has
> committed to acting on it.**

Everything below follows from that sentence.

---

## 1. What counts as a REQUEST

A request is an **instruction directed at the other person** to do something now.

| message | request? | why |
|---|---|---|
| `venmo me 20` | ✅ | instruction |
| `paytm 1000` | ✅ | instruction, terse but unambiguous |
| `book me a cab` | ✅ | instruction |
| `can you send me 500` | ✅ | instruction phrased politely |
| **`you still owe me 500`** | ❌ | **a statement of fact. Nothing is asked.** |
| `how much do i owe you` | ❌ | a question about an amount |
| `wanna split an uber?` | ❌ | a proposal. No action requested of anyone |
| `rent is due next week` | ❌ | statement |

A message that is not a request **must not create a pending**. This is the most
common source of false fires: a statement gets stored, then any warm reply fires it.

## 2. What counts as ACCEPTANCE

Acceptance is the other person **committing to do the thing**. It does not have to be
elaborate — it has to be a commitment.

**Judge the whole message, not its opening.** An apology, an excuse or a complaint
attached to a commitment does not cancel the commitment. Look for whether the person
has agreed to act.

A bare acceptance of a **real request** is a commitment — the request already said what
the action is, so `"sure"` means "yes, I will do that". It does not need to restate it.

| after a real request | fires? | why |
|---|---|---|
| `k` / `ok` / `sure` / `bet` | ✅ | terse, but a complete acceptance |
| `ok sending` / `sending now` | ✅ | commitment + action |
| `on it` / `doing it now` | ✅ | commitment to act |
| `yeah one sec` | ✅ | commitment |
| **`yeah my bad, sending now`** | ✅ | **apology + commitment — the commitment wins** |
| **`sorry forgot, doing it rn`** | ✅ | same shape |
| **`ugh fine, sending`** | ✅ | reluctance is still agreement |
| **`yeah my bad`** | ❌ | **acknowledges fault, commits to nothing** |
| `oh right, i forgot` | ❌ | acknowledgement only |
| `for what?` | ❌ | question — pending stays open |
| `nah im broke` | ❌ | rejection — pending closes |
| `ill pay you friday` | ❌ | future promise — becomes a reminder, not a payment |

The distinction is only ever: **did they commit to doing it, or merely react to being
asked?** Tone, apology and reluctance are all irrelevant.

## 3. Self-initiated — fires immediately, no request needed

| message | fires? |
|---|---|
| `im sending you 20 now` | ✅ money |
| `im booking an uber` | ✅ ride |
| **`just sent it on gpay`** | ❌ **already done — see §3a** |
| `ill pay you back tomorrow` | ❌ future, not now |

## 3a. Completed actions never fire — REVISED 2026-08-02

**The test is: is an action still needed?** Not what the message looks like, not how
recent it is, not whether the request came first. If the thing is already done, the app
has nothing to do and must stay silent — the message is information, not a trigger.

This **reverses** the earlier ruling that `just sent it on gpay` fires. That row shipped
into the training data, so the model currently scores these at 0.996 and is behaving
exactly as it was taught. Correcting it requires regenerating the training windows and
retraining — relabelling the eval alone will not move it.

| message | fires? | why |
|---|---|---|
| `sending now` / `im venmoing you 170 rn` | ✅ | in progress — the action is happening |
| `on it` / `doing it right now` | ✅ | committed, not yet done |
| **`just sent`** / `sent it just now, refresh` | ❌ | **done. nothing to act on** |
| **`transferred check ur app`** | ❌ | done |
| **`cab booked`** / `ok requested, silver civic` | ❌ | done, even though it answers the request |
| `already booked the ola like 10 mins back` | ❌ | done, and not recent |
| `uber already scheduled for 4am` | ❌ | scheduled — nothing to do now |
| **`one sec opening the app`** | ❌ | **narrating preparation, not agreeing — §3b** |
| `lemme grab my phone` | ❌ | same |

Recency is irrelevant. `just sent` and `sent last week` are the same case: done.

## 3b. Preparation is not agreement

Describing what you are about to do is information. Only an explicit agreement, or an
action stated as in progress, fires.

| message | fires? |
|---|---|
| `sure` / `ok` / `yeah` / `bet` (after a real request) | ✅ agreement |
| `sure, one sec opening the app` | ✅ the `sure` carries it |
| `one sec opening the app` | ❌ no agreement, no action yet |
| `one sec pulling it up` | ❌ same |

## 4. Ride means ride-hailing ONLY

| message | fires ride? |
|---|---|
| `book me an uber / ola / cab` | ✅ |
| `can you pick me up?` | ❌ their own car — there is nothing to book |
| `can you drop me home?` | ❌ same |
| `uber is so expensive now` | ❌ discussion |
| `my uber is 5 min away` | ❌ status update on an existing trip |
| `ordering from uber eats` | ❌ food delivery |
| `book my flight ticket` | ❌ not a ride |

## 5. Money means person-to-person transfer ONLY

Not price questions, not bill discussion, not paying a shop or landlord, not future
promises, not idioms (`pay attention`), not jokes (`send me a million lol`).

## 6. Timing and conversation flow

* **Distance does not matter.** A genuine acceptance ten messages later still fires.
  What matters is whether it is genuinely responding to the request.
* **A clarifying question keeps the pending open.** It is not a rejection.
* **Each pending fires at most once.** After it fires it is consumed; a later
  unrelated `sure` must not fire it again.
* **The requester cannot accept their own request.** Repeating or chasing it is not
  acceptance.
* **Two pendings, one reply**: the reply resolves the one it actually addresses. A
  single turn should not fire two intents unless the message genuinely does both.

## 6a. Group chats — anyone present may accept

In a group the request is not aimed at one named person. **Whoever commits, fires.**

```
A: can someone venmo me 20 for the pizza      request stored
C: ok i got you                               FIRE money   C committed
B: oh i would have                            nothing      already consumed
```

The requester still cannot accept their own request.

**Two open requests, one ambiguous reply.** REVISED 2026-08-02 — previously this fired
nothing. It now resolves to the **most recent** open request.

The verb disambiguates whenever it is present: `ok sending` is money, `ok booking` is
ride, regardless of order. Only when there is no verb does recency decide.

```
A: venmo me 20 for pizza          request stored (A)
C: also book me a cab             request stored (C)
B: ok sending                     verb wins  -> FIRE money
B: ok booking it                  verb wins  -> FIRE ride
B: ok                             no verb    -> FIRE ride (C's, most recent)
B: ok [reply_to A's message]      reply_to overrides recency -> FIRE money
```

⚠️ Two caveats on the recency rule, both real:

1. The classifier can only see the last **6 messages**. If the most recent request
   scrolled out of that window it cannot apply this rule at all — it has no memory of
   the request. See the far-request gap (9.2% of fire events).
2. This deliberately fires on ambiguity, which trades precision for recall. It is the
   one ruling that pushes false fires **up** rather than down.

## 6b. The requester may cancel

A request withdrawn by the person who made it is gone. Nothing later can revive it.

```
A: venmo me 20                 request stored
A: actually nvm i found cash   pending CLEARED by the requester
B: ok                          nothing — there is nothing to accept
```

## 6c. Chasing is not a new request

Following up on an unanswered or already-settled request is **the same request**, not
a fresh one. It must not create a second pending or fire a second time.

```
A: venmo me 20                 request stored
B: ok                          FIRE money           consumed
A: did you send it?            chasing — NOT a new request
B: yeah just now               nothing — already fired

A: venmo me 20                 request stored
A: hello?? venmo me            chasing — still ONE pending, not two
B: ok sending                  FIRE money once
```

## 6c2. A rejection is dormant, not dead

People change their minds. A refused request can still be revived — by the person who
refused it later committing, or by the asker asking again and being accepted.

```
A: venmo me 20                 request stored
B: nah im broke                rejected — dormant, nothing fires
B: actually ok fine, sending   FIRE money   B revived it themselves

A: venmo me 20                 request stored
B: nah not right now           dormant
A: come on i really need it    asker pressing — still ONE request
B: ugh fine sending            FIRE money

A: venmo me 20                 request stored
B: nah im broke                dormant
A: ok no worries               nothing
B: how was your weekend        nothing — unrelated, stays dormant
```

Reviving needs the same standard as any acceptance: **a commitment to act now.** A
warm or agreeable message about something else does not revive a dead request. And it
still fires only once.

## 6d. Conditional agreement does not fire

Agreement contingent on something else is not a commitment to act now. Same test as a
future promise: **are they saying they are doing it now?**

| reply to `venmo me 20` | fires? |
|---|---|
| `ok if you cover dinner friday` | ❌ no commitment to send now |
| `sure but only if you pay me back` | ❌ |
| `ok ill send it now if you cover dinner` | ✅ commits to sending now |

## 6e. A counter-offer is a NEW proposal

Agreeing to a *different* amount or trip is not accepting the original. It replaces the
request, and fires only once the other person agrees to the new terms.

```
A: venmo me 50                 request stored
B: i can only do 20            counter-offer — NOT acceptance, nothing fires
A: yeah that works             A agrees -> FIRE money (B is the payer)
```

```
A: venmo me 50                 request stored
B: i can only do 20            counter-offer
A: nah i need the full 50      no agreement -> nothing fires
```

## 6f. Direction — who sees the prompt

Firing correctly is only half the job: the prompt must reach the right person. Both of
these fire money, but the payer is different.

| message | who pays | prompt shown to |
|---|---|---|
| `venmo me 20` (A to B) | **B** | B |
| `im sending you 20` (A to B) | **A** | A |
| `can someone venmo me 20` (group) | whoever accepted | the accepter |
| counter-offer accepted | the person who offered to send | that person |

Rule: **the prompt goes to the person who will part with the money, or who will book
the ride.** Never to the person who asked.

> Status: this is currently decided by separate targeting logic that has never been
> tested against these cases. It needs its own verification pass — firing at the right
> moment for the wrong person is still a broken feature.

## 7. Cost balance

False fires are worse than misses — **but misses are not cheap.**

* A prompt that should not appear: annoying, user dismisses it.
* A prompt that never appears: the feature silently did nothing.

Concretely: precision weighted roughly **2×** recall. Not a hard precision floor
(that buys precision at any price), and not equal weighting.

* threshold selection → maximise **F-beta, beta = 0.7**
* training loss → asymmetric but moderate (`gamma_neg ≈ 2.5`, not 4.0)

---

## Worked examples

```
A: paytm 1000                          request stored
B: k                                   FIRE money           terse, but a real acceptance
                                                            the request already said what
                                                            to do — "k" agrees to it

A: you still owe me 500                NOT a request — no pending
B: yeah my bad                         nothing              nothing to accept

A: venmo me 500                        request stored
B: yeah my bad                         nothing              reaction only
B: ok sending                          FIRE money           commitment

A: venmo me 500                        request stored
B: yeah my bad, sending now            FIRE money           apology + commitment
                                                            the apology changes nothing

A: venmo me 500                        request stored
B: ugh fine, sending                   FIRE money           reluctance is still agreement

A: venmo me 20                         request stored
B: for what?                           nothing, still pending
A: lunch                               nothing
B: ok fine sending                     FIRE money           distance is irrelevant

A: book me a cab                       request stored
B: sure                                FIRE ride
A: also add a reminder                 reminder is unmanaged
B: sure                                nothing for ride     already consumed

A: can u pick me up?                   NOT ride-hailing — no pending
B: sure be there in 10                 nothing
```
