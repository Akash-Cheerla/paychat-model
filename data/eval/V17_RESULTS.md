# v17 + the bare-reply guards vs PRODUCTION (v14 on git HEAD's app.py), 2026-09-17

Candidate: `conv_model_v17` at thresholds **money 0.985 / ride 0.985** (V17_THRESHOLD_PREREG.md -
picked from the validation curve before any log was replayed) on today's app.py, which adds the
2026-09-17 guards. Reference: v14 + `git show HEAD:app.py` (7b172f0), i.e. what users have.

## Real traffic - the deciding numbers

| log | production (v14 + HEAD app.py) | candidate (v17 + guards) |
|---|---|---|
| dev 09-10, adjudicated | 59/82 caught 72.0%, 21/155 wrong 13.5% | **64/82 78.0%, 16/161 wrong 9.9%** |
| **blind holdout 08-11** | 33/40 caught 82.5%, 15/91 wrong 16.5% | **35/40 87.5%, 14/96 wrong 14.6%** |
| 09-11 rooms, adjudicated | 5/6 caught, 4/101 wrong 4.0% | 5/6 caught, 4/100 wrong 4.0% (tie) |
| robustness suite through the server (845 cases, exact labels) | 64.2 fire / 95.3 quiet / 71.4 held-out | **92.5 / 100.0 / 88.5** |

`tests/ship_gate.py --name <arm>`: candidate SHIP, production DO NOT SHIP (it is the baseline).
The guards changed no adjudicated log number: v17 scored the same with and without them.

## Unlabelled dogfood days (08-21, 08-30, 08-31, 09-02, 09-11), 8,926 messages, 0 errors

544 prompts from production, 594 from the candidate, 68 unique differences, read one by one:

- candidate fires 38 that production misses: ~30 are real commitments ("Can you book cab for me" /
  "Sure", "Can you give me 10₹" / "Ok", "can you spot 200 till then?" / "Yep fr", "Sure, How much
  do you need?", "Booking it now" after a failure, "Lemme book it"); ~8 are wrong - the team
  discussing the app (3), a duplicate for one booking (2), the requester's own "Ok" (1), one ride
  prompt on a money thread (1), one quoted-rules example (the known SS3c gap)
- production fires 30 the candidate does not: ~20 are production's own false prompts (meta
  chatter, a re-sent request, the requester's own "Yes"); ~9 are genuine candidate misses, mostly
  just under the line ("Cool" 0.97, "Ok i am transferring" 0.98, "Sending it now" 0.79, "Sure" 0.20)

Latency is unchanged (fp32 ONNX both): median 481 ms production, 515 ms candidate under parallel load.

## Batteries (tests/run_all.py) - candidate wins 5, loses 1, ties 12

| battery | production | candidate |
|---|---|---|
| test_bare_ack_guards (new) | 9/12 | **13/15** |
| test_basics | 80/82 | **81/82** |
| test_reported_bugs | 14/16 | **15/16** |
| test_request_variations | 38/42 | **39/42** |
| test_ride_slots | 14/15 | **15/15** |
| test_self_initiated | **19/22** | 17/22 |
| 82-case context/group set | **75/82** (73/80 excl. known gaps) | 74/82 (**74/80** excl. known gaps) |

## What the guards do (app.py, 2026-09-17)

Both stop a BARE reply that has nothing to agree to; neither can add a prompt.

- ruling 6/7: a third person's bare ok after someone clearly TOOK a group request
  ("let me book a cab", "i'll send it", "booking it now"). A bare "sure" after a bare "sure" still
  fires, clear words from a third person still fire, splits are excepted, and reactions in between
  ("lol", "ok thanks") are skipped - without that skip the guard missed the case in test_stale_slots
  where a bystander's prompt then swallowed the payer's real one.
- s1a: a bare yes after a shortfall that asks for nothing ("im short 300 this month" / "sure").
  Debts are excluded, so "you owe me 20" / "sure" still fires (ruling 8).

## Known, accepted losses (v17 model misses, not the guards)

"i'll send it" 0.972, "sending now" 0.985, "I'll be sending you that dollar that I owe you" 0.141,
"you owe me 20$" / "sure" 0.271, "let me book a cab" answering a group request 0.218, "ok sending"
with two requests open 0.954. These are what v17b is for; a CPU probe showed the current v17b data
is not yet better than v17's (it weakens money acceptances), so v17b needs matched-pair data first.
