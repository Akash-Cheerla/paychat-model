# Conversation classifier — deploy notes

**Status: off by default. Deploying this changes nothing until the flag is set.**

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
```

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

| | rules (today) | classifier |
|---|---|---|
| DeepSeek eval, 2,927 conversations | 57.6% | **81.3%** |
| Claude eval, 400 conversations | 54.8% | **88.2%** |
| long multi-intent eval, 319 conversations | 28.2% | **64.3%** |
| hand-written chats, 21 | 16/20 | **21/21** |
| false fires (long eval) | 149 | **20** |
| missed (long eval) | 207 | **103** |

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

* **Two requests open at once, answered with a bare "ok"** — nothing fires. There is no
  signal for which request is meant, and guessing risks a wrong payment prompt. Naming the
  action ("ok sending" / "ok booking") resolves it correctly. The rule layer scores 6%
  here; the classifier 12%. Neither solves it.
* **Unusual acceptance phrasing is brittle.** `its on the way` fires at 0.97 after a money
  request; `on its way` scores 0.27. Same meaning, same context. Context disambiguation is
  solid — the same phrase scores 0.03 after a gift or parcel question — but surface form
  still swings the score more than it should.
* **Every number above comes from generated conversations.** No real user data has been
  measured. That is what the dogfood logging is for.

## Dogfood logging

Off unless explicitly enabled, and it refuses to start without an expiry date.

```bash
export PAYCHAT_LOG_ALL=1              # or PAYCHAT_LOG_ROOMS=dm_1_2,dm_3_4
export PAYCHAT_LOG_UNTIL=2026-08-10   # required
```

Appends one JSONL line per message — text, sender, room, what fired, and the model's
scores. `PAYCHAT_LOG_ALL` is appropriate only while the product is used solely by people
who have agreed to it; switch to the room list the moment anyone else has an account.
