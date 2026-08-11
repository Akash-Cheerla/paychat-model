# Conversation classifier — deploy notes

**Status: ON in production as of 2026-08-06.**

The code default is still off (`Dockerfile` ships `PAYCHAT_CONV_CLASSIFIER=0`), but the
running deployment sets it to `1`, so the classifier — not `conversation.py` — is what
decides money and ride for real users today. Verified from the dogfood logs: group rooms
fire without `reply_to`, which the rule layer cannot do.

Do not read the rule-layer behaviour in `API_BACKEND.md` as describing production.

## What it is

A second model that decides whether **money** or **ride** should fire, by reading the
last 10 messages of the conversation instead of judging one message in isolation. It
replaces the rule-based decision layer (`conversation.py`) for those two intents only.
The other seven intents are untouched and still come from the main model.

## Turning it on

```bash
export PAYCHAT_CONV_CLASSIFIER=1
```

That is the whole switch. Unset or `0` and the server behaves exactly as it does today.
Rollback is unsetting it — the classifier keeps no state, so there is nothing to migrate
or clean up.

Optional overrides:

```bash
export PAYCHAT_CONV_MODEL=/path/to/conv_model      # default: ./conv_model
export PAYCHAT_CONV_CONTEXT_TTL=14400              # default: 4h, see "timing" below
export PAYCHAT_RIDE_DIVISIBLE=0                    # default: 1, see below
```

`PAYCHAT_RIDE_DIVISIBLE` controls whether a ride request stays answerable after the
first person commits. On (the default), "we need 2 cabs to the airport" / "i'll book
one" / "i'll book the other" gives **both** bookers a prompt carrying the destination;
off, the first commitment consumes the request and everyone after it gets a blank
destination to retype. Off is the pre-2026-08-11 behaviour, kept as a rollback that
needs no redeploy. Money is unaffected either way — it is divisible only when the
wording says so ("1000 each").

## ⚠ The Dockerfile default disagrees with production

`Dockerfile` ships `PAYCHAT_CONV_CLASSIFIER=0`; production runs `1`. A clean rebuild
that does not set the variable in its own environment will come up on the **rule layer**
— a materially different product, and none of the measurements in this repo describe it.
If you are deploying fresh, set it explicitly. This mismatch should be fixed at the
source; it is recorded here because it has already caused one round of confusion.

## What it needs

* `conv_model/` — ships in this commit (git-lfs). `model.safetensors`,
  `conv_model_int8.onnx`, `model_info.json`, tokenizer files.
* `onnxruntime` — in `requirements.txt`. Without it the server silently falls back to
  fp32 PyTorch, which works but is ~4x slower.

It fails loudly on startup if the flag is set and the model cannot load, rather than
silently reverting to the rules. A silent fallback would look like "the new model made
no difference", which is the hardest kind of bug to notice.

## Cost

**13 ms per message** on CPU (INT8 ONNX), on top of the existing intent model. Measured,
not estimated.

## Measured behaviour

Both eval sets replay whole conversations through the real server, message by message, in
a fresh room per conversation. Scored under `FIRING_RULE.md`.

> **The rules-vs-classifier numbers that used to sit here are gone.** They were scored
> before the 2026-08-05 relabel and are not comparable to anything measured since. See
> "A note on comparing numbers" below.

### v5 vs v4 — both re-scored 2026-08-06 on current labels, same server code

| eval set | group content | v4 | v5 |
|---|---|---|---|
| Claude eval, 400 conversations | 0% | 91.8% | **92.2%** |
| long multi-intent eval, 319 | 19.7% | 64.6% | **69.0%** |
| group held-out, 138 | 100%, 3 speakers | 71.0% | **74.6%** |
| group_eval_v6, 787 | 100%, 3-6 speakers | 60.4% | **68.4%** |
| DeepSeek eval, 2,927 conversations | 2.6% | not re-run | 87.4% |
| hand-written chats, 21 | — | — | **21/21** (51/51 turns) |

Conversation accuracy: a conversation counts only if every turn fired exactly right.

The gain scales with group density — flat where there are no groups, largest where
speaker counts are highest, which is what the v5 training round was for. Paired at turn
level on `group_eval_v6`, v5 fixes 150 turns and breaks 70.

### A note on comparing numbers

Eval labels were regenerated **2026-08-05** under `FIRING_RULE.md` §3a (completed actions
no longer fire), moving 36 turns on the Claude set and 226 on the DeepSeek set. Every
`RUN_*.json` older than that date was scored on labels that no longer exist.

This is not academic: v4 scores 89.25% on the Claude eval under the old labels and 91.8%
under the current ones — a 2.55pp gain from the relabel alone, with no model change.
**Re-run the old model, never quote its stored number.** `conv_model_v4/` is kept for
exactly this.

Also hand-labelled live with the product owner across 15 conversations: **11 matched**.
Of the four that did not, two were spec questions rather than model errors, one was a bug
now fixed, and one is a known phrasing gap (below).

## Behaviour changes worth knowing BEFORE you flip it

These will look like bugs if nobody warns the mobile team.

1. **Group chats start firing.** Today nothing fires in a group unless the responder
   swipe-replies. The classifier reads the window and needs no `reply_to`, and no
   `dm_<lo>_<hi>` room-id convention either.

2. **`status` only ever returns `fired` or `no_fire`.** `pending`, `cancelled` and
   `reminder` are rule-layer states the classifier does not model. Anything gating on
   `status == "fired"` is unaffected; anything *reacting* to the other three goes quiet.
   `conversation_state.decided_by == "conv_classifier"` tells you which path ran.

3. **Completed actions no longer fire.** `just sent`, `cab booked`,
   `transferred check ur app` used to produce a prompt and now do not — the action is
   done, so there is nothing to prompt for. See `FIRING_RULE.md` §3a.

## Timing — read this before dogfooding

The classifier's memory is the conversation window, and that window is time-limited by
`PAYCHAT_CONV_CONTEXT_TTL` (default **4 hours**, matching `PENDING_TTL` on the rule path).

This was set to 5 minutes and it was a serious bug: a reply seven minutes after the
request saw an EMPTY window and scored 0.03 instead of 0.997. Every eval replayed messages
milliseconds apart, so no test caught it. Verified fixed with a replay that sleeps for
real between turns — requests at 7 min, 45 min and 3 hours all still fire, and 5 hours
correctly does not.

**Known limitation: context does not survive a restart.** `ConversationContext` is
in-memory. After a deploy or a crash, a reply whose request arrived beforehand sees an
empty window and will not fire — regardless of the TTL. Confirmed: 0.997 before restart,
0.033 after. This is pre-existing, not introduced by the classifier, but it matters more
now that the window IS the memory. During a dogfood week, a mid-week deploy will produce
"failures" that are actually this. Fixing it properly means persisting the context (Redis)
or rehydrating from the backend's message history.

## Known gaps

* **Two requests open at once, answered with a bare "ok"** — the classifier answers the
  most recent request. Confirmed as the wanted behaviour: prompts render as persistent
  chips under the message rather than modal sheets, so a wrong guess costs a tap, while
  firing nothing leaves the most common reply in chat with no prompt at all. Naming the
  action ("ok sending" / "ok booking") is unambiguous either way. The rule layer still
  fires nothing here.
* **Unusual acceptance phrasing is brittle.** `its on the way` fires at 0.97 after a money
  request; `on its way` scores 0.27. Same meaning, same context. Context disambiguation is
  solid — the same phrase scores 0.03 after a gift or parcel question — but surface form
  still swings the score more than it should.
* **"your place" resolves against the wrong speaker.** "im stuck at office" ->
  "ill book you an uber to your place" yields `destination: Office` — it resolves the
  vague place from the window without tracking whose place it is. Needs speaker-aware
  resolution.
* **`let me <app> you <amount>` is a hole.** `let me send/transfer/pay/venmo/upi/phonepe
  you 100` all fire; `let me gpay/paypal/cashapp/zelle/paytm you 100` and `let me give you
  100` score 0.03-0.47. Every *other* phrasing of those same apps is fine — `im gpaying
  you 100`, `sending you 100 on gpay`, `ill cashapp you the 100 now`, `sure sending on
  zelle` all fire at 1.00. So it is one sentence frame, not missing app coverage. Present
  in v4 too, which fires on **none** of that frame, so v5 is strictly better here.
  Notable because gpay, paypal, cashapp and zelle are the four apps the product supports.
* **An offer can fire twice.** `shall I send you 500?` fires on the question (targeted at
  the sender, who is the payer — correct per FIRING_RULE §6e), and the acceptance can fire
  again: two prompts, one payment. v4 fires on neither the question nor the acceptance,
  so this is a new over-firing path rather than a regression.
* **Every number above comes from generated conversations.** No real user data has been
  measured yet. The 2026-08-04/06 dogfood logs could not be used — the iOS and Android
  clients had delivery bugs and had not implemented group intents at all, so the logs
  reflect client behaviour more than model behaviour.

## Slot freshness

`conversation_state.triggered_by.slots` carries the **effective** values, not a snapshot
of the request. Three cases where those differ, all now handled:

| conversation | triggered_by.slots |
|---|---|
| "lend me 2000" -> "i can only do 1000" -> "cool sending now" | `amount: $1000` |
| "cab to koramangala" -> "actually make it indiranagar" -> "ok booking" | `destination: Indiranagar` |
| "get me a cab" -> "where to?" -> "whitefield" -> "ok booking" | `destination: Whitefield` |

`triggered_by.text` still holds the original wording. Clients must read the slots — the
amount parsed out of the text is the superseded one, which on a payment sheet pre-fills
double what was agreed.

## Dogfood logging

Off unless explicitly enabled, and it refuses to start without an expiry date.

```bash
export PAYCHAT_LOG_ALL=1              # or PAYCHAT_LOG_ROOMS=dm_1_2,dm_3_4
export PAYCHAT_LOG_UNTIL=2026-08-10   # required
```

Appends one JSONL line per message — text, sender, room, what fired, and the model's
scores. `PAYCHAT_LOG_ALL` is appropriate only while the product is used solely by people
who have agreed to it; switch to the room list the moment anyone else has an account.
