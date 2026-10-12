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
3. The bullet list is **variable length**, typically 2–9. It is not a design constant — it
   is however many things the message contains.

Nothing else. A card may end with a bullet reading `Tickets: QA-101, QA-102`, or one
reading `Build: https://install.shareipa.com/ccUH8I` / `Design: https://figma.com/...` —
those are ordinary bullets and need no special case on either platform. If the client
already linkifies URLs in text, they will be tappable for free; if not, they are still
readable and that is enough.

---

## What it does not do, and you should know before anyone asks

**It summarises one message, not a conversation.** That reaches the long messages and
nothing else. On real traffic another ~10% of messages carry a deadline, an owner or a
decision while being too short to qualify — so a decision reached across five short
replies is invisible to this.

A live rehearsal of a group day made that concrete: a 500 on an endpoint was reported,
diagnosed and fixed across four short messages, and appears on no card. Neither does a
QA ticket still failing — "QA-116 still fails" is four words, so the gate rejects it and
the ticket-number restoration never sees it. Restoring ids only helps inside a message
that already earned a card.

Conversation-level summaries — "what was decided in this room since you last looked" —
are the fix, and are what Zoom and Teams both ship. They need this model to exist first,
which is why per-message goes out first.

**It sometimes garbles a clause.** "Cannot make any other user as admin" came back as
"Cannot make admin as admin" (dogfood 2026-10-10). It does not invent numbers, amounts or
ticket IDs — zero across every card measured, on 69 archived cards and again on a blind
set — so the failure is awkward phrasing rather than wrong facts. Fixing it needs a
retrain on more data of this shape, not another rule; adding rules to patch generation is
how the title logic went wrong once already.

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

**Links are content.** The gate used to delete URLs before measuring length, so "please
use the following build for testing `<url>` 1. the group profile image issues have been
addressed" measured 124 characters, failed the 140 bar and produced no card — one of the
most useful messages in the room. Links are now shortened rather than removed, so the
length is what a reader would actually read (that message measures 149), and a URL in the
message is appended to the card as its own labelled bullet. A message that really is
nothing but links still fails, and a described link only earns a card when it points at
something the team works in — a Figma, a doc, a ticket, a build. A blog link in "no action
needed I just think you'll enjoy it" does not.

Three scoring gaps closed at the same time, all found on the 2026-10-10 log:

- `issue` could not match **"issues"**, which is the commonest bug word in these rooms —
  "the group profile image issues and UI issues have been addressed" scored zero. Same for
  `bug`, `error`, `regression`. "addressed" was not read as a fix either.
- A bare **"cannot"** scored nothing, so "1. Cannot set group profiles pictures once
  created 2. Cannot make any other user as admin" produced no card. `unable to` and
  `doesn't work` had always matched.
- **"I wanted …"** was treated as throat-clearing and cancelled the sentence it introduced,
  which deleted "I wanted a call to take a decision on Ram which is 4GB rn but we may need
  8GB" — the most consequential message in the log that day.

**A name is sometimes mistaken for the topic.** "we'll need gpu's Gowtham , just fyi"
comes back titled "Gpu Gowtham Needed" and "Still going on Gowtham !" as "Gowtham Update".

This is a known limitation, not a fix, and deliberately so. Stripping the addressed name
before the model reads the message fixed both titles and was then measured: it touched 57
of 90 generated messages and took production recall from 62.8% to 61.8%, because a rule
that removes a capitalised word sitting against a comma cannot tell a person from a
product, and in a work chat the person addressed is usually the person who now owns the
thing. It was deleting

```
reproduced it three times on Chrome, and once on Safari
Add the index and test on staging (owner: Mike, due: tomorrow)
Mike, you're on deployment
```

Losing who owns a task is worse than an odd title, so the rule was withdrawn. Doing it
properly needs the room's participant names from the backend so there is no guessing -
and even then `owner: Mike,` has to survive. A greeting needs no handling either way:
"Hi Gowtham , I wanted a call to take a decision on Ram" gives "Decision On Ram" with or
without it.

**Ticket numbers are put back.** The student invents no identifiers — 0 fabricated across
69 real cards — but it used to omit them, and ticket numbers are the single thing most
worth keeping. Measured over the tokens the labels mark important:

| token | before | after |
|---|---|---|
| **ticket id** | **8/20 — 40.0%** | **20/20 — 100%** |
| version | 19/24 — 79.2% | unchanged |
| weekday | 65/77 — 84.4% | unchanged |
| clock time | 29/35 — 82.9% | unchanged |
| ratio/size | 14/16 — 87.5% | unchanged |
| amount | 3/3 — 100% | unchanged |
| **all** | 138/175 — 78.9% | **150/175 — 85.7%** |

An id already in the message is restored two ways. Where a bullet clearly came from the
sentence carrying it, the id is appended to that bullet — `Crash on iOS 17.2 when applying
gift card (JIRA-4471)`. Where the id sits in a sentence with no content of its own ("That's
PAY-1198", "logged all of these under QA-101, QA-102, QA-103"), no bullet can overlap it
and guessing a neighbour would risk naming the wrong ticket, so it goes on one trailing
line: `Tickets: QA-101, QA-102, QA-103`.

It never invents, never attaches an id to a bullet from a different sentence, never reads
a version or a date (`3.2.0-rc3`, `2026-10-18`, `4.5:1`) as a ticket, and adds at most
that one line. `tests/test_identifiers.py` holds all of it, because a misattributed ticket
number would be worse than the omission this fixes. On a realistic 31-message day it
produced **zero** trailing lines — it stays dormant unless an id would otherwise be lost.

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
