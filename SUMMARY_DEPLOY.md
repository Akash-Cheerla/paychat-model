# Message summaries — what to deploy and what it does

The Summary tab on the Actions screen. iOS does this on-device today and Android does not
do it at all, so the two platforms cannot agree and a wording fix needs an App Store
release. This moves it to the server, behind one endpoint, so both apps render the same
card and fixes ship without a submission.

---

## Deploying

This is a **second service**, not a change to the intent API. Same repo, same image,
different process — see "Why a separate process" below, it is not a preference.

```bash
git pull                      # brings summary_model/ via LFS, ~950MB
git lfs pull                  # if your clone was shallow
pip install -r requirements.txt

# the intent API is unchanged and keeps its own process
python -m uvicorn app:app         --host 0.0.0.0 --port 8000

# the summary service, separately, and it can live on a different box
PAYCHAT_SUMMARY_THREADS=2 python -m uvicorn summary_app:app --host 0.0.0.0 --port 8001

curl localhost:8001/summary-status
```

From Docker it is the same image with the CMD overridden:

```bash
docker run ... <image> python -m uvicorn summary_app:app --host 0.0.0.0 --port 8001
```

`/message-summary` is **not** served by the intent API any more — it returns 404 there.

`/summary-status` is the check that matters:

```json
{"student_loaded": true, "serving": "student+rules",
 "threads": 2, "threads_live": null, "weights_resident": false}
```

`threads` is what the model will use; `threads_live` and `weights_resident` are null and
false until the first card forces the lazy load, which is normal on a fresh box.

If it says `"rules only (weights not found)"` the endpoint still works — it serves the
rules card instead of the model's. That is deliberate: a missing weights file degrades the
feature rather than breaking the service. It also means a failed LFS pull is easy to miss,
which is why this endpoint exists.

### What it needs

| | |
|---|---|
| memory | **~1.4 GB peak** (weights are 443 MB; beam search on a long input is the rest) |
| CPU | 2 threads. More is slower — measured 4.0s/card on 4 threads, 2.8s on 2 |
| GPU | none |
| disk | ~950 MB for `summary_model/` |

Two threads is pinned deliberately, so the other vCPUs stay free for `/classify`, which
is the path that has to stay at 69ms. Separate processes stop the summary blocking that
path; they cannot stop it competing for cores, so the thread cap is still what protects
it. Measured on 4 cores with `/classify` under continuous load: 96ms idle, 273ms while a
card is being written.

### Environment

| variable | default | |
|---|---|---|
| `PAYCHAT_SUMMARY_MODEL` | `summary_model/` | elsewhere if the weights live off-repo |
| `PAYCHAT_SUMMARY_THREADS` | `2` | raising it made things slower in testing |
| `PAYCHAT_SUMMARY_BATCH_CALLS` | `12` | model calls allowed in one batch request |

---

## The API

### One message

```
POST /message-summary
{"text": "Build 52 assigned in live. Please test the entirety of the app. ..."}
```

```json
{
  "summary": {
    "title": "Build 52",
    "points": ["Build 52 assigned in live",
               "Test entirety of app",
               "Marked stable release",
               "Ensure 0 issues in positive flow"]
  },
  "source": "student+rules",
  "as_of": "2026-10-11T04:12:09Z"
}
```

**`summary` is `null` for about 99% of messages, and that is correct.** A gate runs before
the model: 140+ characters and at least two sentences. "ok", "Linux os", "pulled it" never
reach it — a summary of a two-word message is longer than the message. On our dogfood log,
5,190 real messages produced 151 that qualified and 69 cards.

Null returns in **~0ms**. The gate rejects before the model does any work, so calling this
on every message is cheap.

### A batch

```
POST /message-summary
{"messages": [{"message_id": "m1", "text": "..."}, ...]}      # up to 100
```

```json
{"results": [{"message_id": "m1", "summary": {...}, "source": "student+rules"}],
 "model_calls_used": 2, "as_of": "..."}
```

At most **12 messages per request** reach the model (`PAYCHAT_SUMMARY_BATCH_CALLS`). Past
that the rules answer instead — instantly. A card costs 3–10s, so an uncapped batch of
100 long messages would be a ten-minute request. In practice the gate rejects ~99% and a batch
of 100 needs one or two model calls, so the cap is never reached; `model_calls_used` tells
you if it ever is.

### Timing

| | |
|---|---|
| short message (~99% of traffic) | **~0ms** |
| a card, short end | **~3s** |
| a card, a long status update | **5–10s** (median 6.4, max 9.6 over a 31-message day at 2 threads) |

Summarising runs **outside** the `/classify` request. Call it when a message arrives, store
the card against the message, serve it to the client on request. Do not block anything on
it — the card appearing a few seconds after the message is fine; the intent path slowing
down is not.

---

## For the apps

Three things, and only three:

1. Render a card: a title and a list of bullets.
2. If the stored summary is `null`, **draw nothing**. No empty card, no placeholder.
3. The bullet list is **variable length**, typically 2–8. It is not a design constant — it
   is however many things the message contains.

---

## What it does not do, and you should know before anyone asks

**It summarises one message, not a conversation.** That reaches the long messages and
nothing else. On real traffic another ~10% of messages carry a deadline, an owner or a
decision while being too short to qualify — so a decision reached across five short
replies is invisible to this.

A live rehearsal of a group day made that concrete: a 500 on an endpoint was reported,
diagnosed and fixed across four short messages, and appears on no card. Neither does a
QA ticket still failing.

Conversation-level summaries — "what was decided in this room since you last looked" —
are the fix, and are what Zoom and Teams both ship. They need this model to exist first,
which is why per-message goes out first.

---

## How it works, briefly

Two halves, each doing what it measurably does better.

**Rules decide whether a card exists.** They drop 90.5% of pure chatter — 19 of 21
all-noise messages in the generated set, and 82 of 151 qualifying dogfood messages.

They used to write the title too, on the strength of one example ("Build 52" against the
model's "App Build Testing"). Counted over 69 cards of real traffic that did not hold:

| | rules | student |
|---|---|---|
| weak title (generic, or one bare word) | **72.9%** | **7.2%** |
| message has an identifier, title keeps it | 1/20 | 1/20 |

The rules won nothing on identifiers and produced "Bug" 15 times and "Build" 11 times
across 70 cards — a tab showing a column of the same word. The student writes the title
now, and invents no numbers in it (0 of 69). The rules title remains the fallback when
the student is absent or fails.

**A distilled model writes the bullets.** flan-t5-base, 250M, fine-tuned on 1,796
(message → card) pairs generated by a larger open-weight teacher on our own hardware. On
5,190 real messages it produces 5.2% verbatim at 51% of source length, with zero invented
numbers across 326 bullets. The rules alone reprinted sentences — 60.7% of their bullets
were verbatim copies — which is what this replaced.

Output passes de-duplication and the structural guards before it is returned: a seq2seq
under beam search restates itself, and 16% of raw bullets on real traffic were repeats or
fragments. None survive.

---

## Why a separate process

Not a preference. The summary ran inside the intent API for one afternoon and that was
wrong in three measurable ways, each of which lands on `/classify`:

| | |
|---|---|
| **the event loop** | A card is 3–10s of blocking torch. Served from `app.py`, one summary took `/classify` from **86ms to 5,571ms** — a single `/classify` completed during one 5.6s summary, because everything else queued behind it. One uvicorn worker, nothing to absorb it. |
| **torch's threads** | `torch.set_num_threads` is process-global. The summary wants 2. The intent model normally runs on ONNX and is untouched — but it has a PyTorch fallback, so if `model.onnx` ever misses an LFS pull, the summary silently halves the intent path's threads. |
| **memory** | The student peaks at 1.4 GB. In one container that counts against the same limit as the intent models, so a long message could push out the path that has to stay up. |

Split, `/classify` holds at **273ms median under continuous load while a card is being
written** (96ms idle) — that residue is CPU contention on 4 cores, which is physics, and
is what the 2-thread cap is for. The summary service can also be moved to another box,
restarted, or killed without the intent API noticing.

Same repo, because they share nothing but a deploy and splitting the repo would double
the LFS store and the test suite for no benefit. Nothing in `summary_app.py` imports the
intent pipeline, and nothing in `app.py` imports the summariser — that is enforceable by
grep and worth keeping that way.
