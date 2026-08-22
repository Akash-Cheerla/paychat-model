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

> **An intent fires only when someone intends to BOOK, SEND, or PAY.
> No such intention, no prompt.**

Product owner, 2026-08-14. Apply this test first, before anything else in this
document. If the answer is no, stop — nothing further can make it fire.

The longer form, which the sections below expand on:

> **Fire only when there is an actionable cause, and the responsible party has
> committed to acting on it.**

Why the short test earns its place at the top: 66 eval labels were found firing `ride`
for a friend agreeing to drive someone in their own car — "ya ill take the car, be ready
by 6:45". Every one of them passes a loose reading of "committed to acting on it", and
every one fails the book/send/pay test in a second. Nobody was booking anything, so
there was nothing for the app to do.

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

## 1a. A statement fires only on an EXPLICIT commitment — added 2026-08-11

§1 says a statement is not a request and creates no pending. That leaves the case that
comes up constantly in real chats: someone states a debt, a shortfall or a cost, and the
other person replies. What happens depends entirely on **what the reply commits to**.

A bare acknowledgement after a statement fires nothing — "alright" or "ok" is most often
just receiving the information. The prompt appears only when the reply says they will
act.

| conversation | fires? |
|---|---|
| `you still owe me 300` / `alright` | ❌ acknowledging the fact |
| `you still owe me 300` / `oh i forgot, will send now` | ✅ commitment |
| `im short 300 this month` / `sure` | ❌ a polite noise, nothing agreed |
| `im short 300 this month` / `im sorry, will send` | ✅ commitment |
| `dinner was 3000 last night` / `ok` | ❌ cost talk |
| `dinner was 3000 last night` / `ill send my half now` | ✅ commitment |
| `need 300 for rent, can you help` / `i can help man, sending now` | ✅ commitment |

**The test:** did the reply say they will send/pay/book, or did it only acknowledge?

This is the one place where a bare `ok` behaves differently after a statement than after
a request. After a REAL request (§2) a bare `ok` is a full acceptance, because the
request already named the action — there is something to say "ok" to. After a statement
there is nothing to accept, so the commitment has to be in the reply itself.

The amount stated is still carried onto the prompt when it does fire. The statement is
not a request, but it is where the figure lives.

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

**Reviewed twice, kept.** On 2026-08-03 this was briefly reversed — shown the message in
a real chat, "one sec opening the app" reads like someone who has clearly decided to pay,
and firing there is arguably more useful. It was reverted the same day for a reason worth
recording, so this is not re-litigated a third time:

The model does not learn "opening the payment app fires". It learns a pattern, and the
nearest neighbours come with it — `one sec`, `lemme check`, `gimme a sec`,
`hold on lemme see`. That is precisely how v2 ended up firing on `hmm` at 0.996, which
cost a full retrain to undo.

The cost of the current rule is narrow: someone who opens the app without saying anything
affirmative gets no prompt on that message, and gets one moments later when they say
`sending`. The cost of the reversal is every `one sec` in every conversation becoming a
payment prompt. Revisit only with real usage data showing bare preparation is common AND
that the false-fire risk did not materialise.

## 3c. Need, offer, action — three shapes that look alike — added 2026-08-20

Ruled by Gowtham over WhatsApp on 2026-08-20. These three read almost identically and
fire completely differently, which is why the training data ended up split roughly
50/50 on `let me book a cab` and the model scores it 0.07 in every context.

| shape | example | fires? |
|---|---|---|
| **need** | `ill need to book a cab back home` | ❌ never — a need is not an action |
| **offer** | `let me book a cab` / `let me send you 500` | ❌ not alone — waits for the other party |
| **in progress** | `im booking an uber` / `im sending you 20 now` | ✅ immediately, per §3 |

`I may book a cab ride home` is a need. `need` is not a trigger word, it is the
opposite of one.

**An offer that ANSWERS a request is an acceptance, and fires at once.** "Request and
answer will always trigger" — so `can you book me a cab to the airport` / `let me book
a cab` fires on the second message, and so does `ok let me book a cab`. The waiting
rule applies only to an offer nobody asked for.

An unprompted offer fires on the confirmation, not on a third message from the offerer:
`let me send you 500` / `ok` fires on the `ok`. The offerer does not have to speak again.

**Who sees it.** The prompt goes to whoever performs the action — the person paying, or
the person booking — never automatically to whoever typed the confirming message. On
`let me send you 500` / `ok`, the prompt is A's, though B sent the message that fired it.

**Everyone who answered gets it.** In a group, if two people both answer the request,
both get the prompt; it is theirs to sort out who acts. Delivered as a list of user IDs,
not as a broadcast to the room.

**No time limit.** An open request stays answerable for as long as it is still in the
window. A request at 9am answered at 6pm fires.

**A prompt can be withdrawn.** An explicit reversal (`maybe not now`, `nvm`) or a report
that it is already done (`already sent it`, `cab's booked`) takes a shown prompt back off
the screen. A withdrawn intent may fire again if the deal is revived — `actually go ahead
send it` prompts a second time, and that is not a duplicate.

### Not yet solved: quoted conversations

On 2026-08-20 a ride prompt appeared while two of us were discussing these very rules.
The message was:

    Let's say
    B: can you book me a cab to airport (around 9 AM)
    A: okay(6-7PM)

That single message scores 0.997 — it contains a request and an acceptance, so the model
reads it as a real exchange. The `Yes` two messages later fired again off the same primed
context. Andril's earlier report was the same category: he described the app and the app
responded.

The self-acknowledgement guard cannot help. The quoted request genuinely came from the
other speaker, so there is a real open request and `Yes` is a real acceptance. Every
component behaved correctly.

No rule here yet — recording it because it is the third instance and the team is
currently the whole user base.

## 3d. A greeting is not an acceptance - added 2026-08-21

From the dogfood log, real team traffic:

    26: Can you send me 10$
    53: Hi                    <- money 0.997, payment sheet opened
    26: ?

| message | fires? |
|---|---|
| `hi` / `hey` / `yo` / `hello` / `good morning` / `sup` | no, whatever came before |
| `hey can you send me 20` | yes - a request, not a greeting |
| `yo im sending now` | yes - the commitment carries it |

Whole-message match only. A greeting with anything else attached is ordinary text and is
read normally.

**This is not the acknowledgement guard and could not be.** In §3a the speaker was
answering himself, so there was nothing to accept. Here the request is genuine, it came
from the other person, and the reply really is a reply to it. What is wrong is the reply.

Worth recording how it behaves, because it explains why nobody caught it: `hi` scores
**0.033** when the window is short and **0.997** once the window holds a few greetings.
The model is not reading the word, it is reading "short reply after a request" as an
acknowledgement. `yo` fires at 0.995 in any context. So this is a model failure held shut
by a rule, and it should be revisited when the training data covers greetings as
non-acceptances.

### Still open on the same battery

Against a live request, these SHOULD fire and do not:

| message | score | threshold |
|---|---|---|
| `okay` | 0.076 | 0.97 |
| `yeah` | 0.356 | 0.97 |
| 👍 | 0.034 | 0.97 |

`ok` fires at 0.994 and `okay` scores 0.076, which is not a distinction any rule should be
asked to defend. The thumbs up matters most - it was named explicitly as a confirmation on
2026-08-20 and the model has almost certainly never seen one in training. All three are
recall and belong in the next round, not in a regex.

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

## 5a. Splits — one request, owed by everyone separately

A split is **not** answered by whoever pays first. "dinner came to 3000, send me your
shares" is owed by every member, so each person who commits gets their own prompt for
their own share. The same person restating their commitment does not get a second one.

The amount shown is the **share**, never the total — and we would rather show nothing
than the wrong figure:

| the message says | prompt shows |
|---|---|
| `dinner was 3000, split 3 ways` | 1000 — headcount is in the message |
| `dinner was 3000, lets split it` + `participants: 3` | 1000 — headcount from the backend |
| `dinner was 3000, lets split it`, no `participants` | **blank** |
| `dinner was 5000, thats 1000 each` | 1000 — already per-person, never divided again |

The blank case is the important one. A payment sheet pre-filled with 3000 for someone
who owes 1000 invites sending triple; an empty field costs them one typed number. In a
DM the headcount is known to be two, so nothing is ever blank there.

`lets split it` with no number is the commonest phrasing of all and must be treated as
a split — for a while it was not, and the full total went onto the sheet.

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
