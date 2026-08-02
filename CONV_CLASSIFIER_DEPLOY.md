# Conversation classifier — deploy notes

**Status: off by default. Deploying this changes nothing until the flag is set.**

## What it is

A second model that decides whether **money** or **ride** should fire, by reading the
last 10 messages of the conversation instead of judging one message in isolation. It
replaces the rule-based decision layer (`conversation.py`) for those two intents
only. Every other intent is untouched.

## Turning it on

```bash
export PAYCHAT_CONV_CLASSIFIER=1
```

That is the whole switch. Unset or `0` and the server behaves exactly as it does today.

Optional override, only needed if the model lives somewhere else:

```bash
export PAYCHAT_CONV_MODEL=/path/to/conv_model
```

Default is `conv_model/` next to `app.py`, not the working directory.

## What it needs

* `conv_model/` — ships in this commit (git-lfs). Contains `model.safetensors`,
  `conv_model_int8.onnx`, `model_info.json`, tokenizer files.
* `onnxruntime` — already a dependency. Without it the server falls back to PyTorch,
  which still works but is ~3.5x slower.

It fails loudly on startup if the flag is set and the model cannot load, rather than
silently reverting to the rules. That is deliberate: a silent fallback would look like
"the new model made no difference".

## Cost

~18ms per message on CPU (INT8 ONNX), on top of the existing intent model. Measured, not
estimated.

## Measured behaviour

Both test sets replay full conversations through the real server, message by message,
in a fresh DM room per conversation. Scored under `FIRING_RULE.md` as revised 2026-08-02.

| | rules (today) | classifier |
|---|---|---|
| DeepSeek eval, 2,927 conversations | 57.6% | **86.9%** |
| Claude eval, 400 conversations | 54.5% | **93.8%** |
| hand-written chats, 21 | 16/20 | **21/21** |
| false fires | 1,271 | **256** |
| missed detections | 473 | **172** |

## Behaviour changes worth knowing before you flip it

1. **Group chats start firing.** Today nothing fires in a group unless the message
   carries `reply_to`, because ambient matching needs a `dm_<lo>_<hi>` room id. The
   classifier reads the window and needs neither. If a group suddenly produces prompts
   after enabling this, that is the fix working, not a bug.

2. **Completed actions no longer fire.** `just sent`, `cab booked`,
   `transferred check ur app` used to produce a prompt and now do not — the action is
   done, so there is nothing for the app to do. See `FIRING_RULE.md` §3a.

3. **`room_id` no longer has to start with `dm_`** for money/ride to work. The rule
   layer still needs it for other intents.

## Rolling back

Unset `PAYCHAT_CONV_CLASSIFIER`. No data migration, no schema change, no state to clean
up — the classifier keeps no state at all.

## Known gaps

* Reluctant agreement is still weak: `nah lemme just do it rn` scores 0.042 and does not
  fire.
* Adversarial test set sits at 62.1%, the weakest area.
* Every number above comes from AI-generated conversations. No real user data has been
  measured yet.
