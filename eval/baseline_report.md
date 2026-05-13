# FYOE v3 — seed baseline report

**Generated:** 2026-05-12 18:10  
**Model:** roberta-base v3 (125M params, device: `cpu`)  
**IID test-set accuracy (training distribution):** 98.57%  
**Seed-suite size:** 125 curated real-world cases  
**Latency:** 53 ms / case (single-batch CPU)

## TL;DR

- **113/125 passed (90.4%)** on the seed suite
- **12 failures** — these become v4 training data
- Reality gap: **8.2 percentage points** between IID test set and adversarial seed suite
- This gap is the work item. Closing it = production-ready model.

## Performance by category

| Tag | Cases | Passed | Pass rate | Status |
|---|---:|---:|---:|---|
| `smoke` | 14 | 10 | 71% | needs work |
| `multi_intent` | 6 | 3 | 50% | needs work |
| `numerics` | 3 | 3 | 100% | PERFECT |
| `slang` | 4 | 4 | 100% | PERFECT |
| `typo` | 1 | 1 | 100% | PERFECT |
| `long` | 1 | 1 | 100% | PERFECT |
| `negation` | 8 | 6 | 75% | needs work |
| `past_tense` | 7 | 7 | 100% | PERFECT |
| `ambiguous` | 4 | 4 | 100% | PERFECT |
| `chitchat` | 6 | 6 | 100% | PERFECT |
| `false_positive_bait` | 7 | 7 | 100% | PERFECT |
| `code_mixed` | 4 | 4 | 100% | PERFECT |
| `edge_pronoun` | 3 | 3 | 100% | PERFECT |
| `edge_negation_marker` | 1 | 0 | 0% | CRITICAL |
| `edge_multi_recipient` | 1 | 1 | 100% | PERFECT |
| `edge_missing_slot` | 1 | 1 | 100% | PERFECT |
| `alarm_with_note` | 1 | 1 | 100% | PERFECT |
| `alarm_missing_note` | 1 | 1 | 100% | PERFECT |
| `alarm_missing_all` | 1 | 1 | 100% | PERFECT |
| `maps_simple` | 2 | 2 | 100% | PERFECT |
| `maps_query_not_action` | 1 | 1 | 100% | PERFECT |
| `money_split` | 4 | 4 | 100% | PERFECT |
| `query_not_action` | 1 | 1 | 100% | PERFECT |
| `health_implicit` | 1 | 1 | 100% | PERFECT |
| `weather_minimal` | 2 | 1 | 50% | needs work |
| `v5_context_buildup` | 3 | 3 | 100% | PERFECT |
| `v5_confirmation` | 3 | 3 | 100% | PERFECT |
| `v5_confirm_negative` | 2 | 2 | 100% | PERFECT |
| `v5_context_negative` | 2 | 2 | 100% | PERFECT |
| `v5_correction` | 1 | 1 | 100% | PERFECT |
| `v5_no_carryforward` | 1 | 1 | 100% | PERFECT |
| `v5_disambiguation` | 2 | 2 | 100% | PERFECT |
| `v5_single_with_context` | 1 | 1 | 100% | PERFECT |
| `question_form` | 5 | 4 | 80% | minor gaps |
| `idiomatic_false_pos` | 8 | 8 | 100% | PERFECT |
| `present_activity` | 3 | 3 | 100% | PERFECT |
| `third_person` | 3 | 3 | 100% | PERFECT |
| `ultra_short` | 3 | 3 | 100% | PERFECT |
| `bare_word` | 3 | 3 | 100% | PERFECT |

## All 12 failures (grouped by category)

### `smoke` — 4 failures

- `remind me to call mom tomorrow at 6pm`  
  expected **alarm, contact** · fired **alarm**, **missed:** contact  
  top probs: `alarm`=0.98 · `contact`=0.82 · `task`=0.15 · `calendar`=0.10 · `maps`=0.07
- `uber to JFK at 5am tomorrow`  
  expected **ride, maps** · fired **alarm, maps, ride**, **extra:** alarm  
  top probs: `maps`=0.97 · `ride`=0.95 · `alarm`=0.79 · `calendar`=0.09 · `task`=0.09
- `movie night, watching Oppenheimer`  
  expected **video** · fired **calendar, video**, **extra:** calendar  
  top probs: `video`=0.93 · `calendar`=0.63 · `reservation`=0.12 · `tickets`=0.06 · `weather`=0.05
- `book a table at Blue Bottle for 4 people friday at 7`  
  expected **reservation** · fired **calendar, reservation**, **extra:** calendar  
  top probs: `reservation`=0.96 · `calendar`=0.78 · `money`=0.07 · `video`=0.07 · `alarm`=0.06

### `multi_intent` — 3 failures

- `remind me to pay rent friday`  
  expected **alarm, bills** · fired **money, alarm, bills**, **extra:** money  
  top probs: `alarm`=0.97 · `money`=0.74 · `bills`=0.60 · `food_order`=0.10 · `contact`=0.08
- `lunch with sarah friday at 1 at sweetgreen`  
  expected **calendar, contact** · fired **calendar, maps**, **extra:** maps, **missed:** contact  
  top probs: `calendar`=0.96 · `contact`=0.82 · `maps`=0.73 · `reservation`=0.07 · `task`=0.07
- `oppenheimer at amc tonight at 9`  
  expected **video, tickets** · fired **maps, video, tickets**, **extra:** maps  
  top probs: `tickets`=0.94 · `video`=0.87 · `maps`=0.51 · `ride`=0.09 · `music`=0.09

### `negation` — 2 failures

- `don't transfer the rent yet`  
  expected **(none / chitchat)** · fired **money, bills**, **extra:** bills, money  
  top probs: `bills`=0.95 · `money`=0.72 · `task`=0.06 · `alarm`=0.05 · `music`=0.05
- `cancel the reminder for 6pm`  
  expected **(none / chitchat)** · fired **alarm**, **extra:** alarm  
  top probs: `alarm`=0.68 · `calendar`=0.08 · `money`=0.02 · `bills`=0.02 · `task`=0.02

### `edge_negation_marker` — 1 failure

- `don't forget to book the flight for next month`  
  expected **travel** · fired **alarm, travel**, **extra:** alarm  
  top probs: `travel`=0.93 · `alarm`=0.81 · `calendar`=0.10 · `health`=0.09 · `task`=0.07

### `weather_minimal` — 1 failure

- `weather`  
  expected **weather** · fired **(none)**, **missed:** weather  
  top probs: `alarm`=0.01 · `weather`=0.01 · `maps`=0.01 · `contact`=0.01 · `food_order`=0.01

### `question_form` — 1 failure

- `wanna uber there`  
  expected **ride** · fired **maps, ride**, **extra:** maps  
  top probs: `ride`=0.98 · `maps`=0.81 · `food_order`=0.09 · `reservation`=0.06 · `task`=0.05

## Per-intent precision / recall (on seed suite)

| Intent | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `money` | 16 | 2 | 0 | 89% | 100% | 94% |
| `alarm` | 10 | 3 | 0 | 77% | 100% | 87% |
| `contact` | 4 | 0 | 2 | 100% | 67% | 80% |
| `calendar` | 4 | 2 | 0 | 67% | 100% | 80% |
| `maps` | 6 | 3 | 0 | 67% | 100% | 80% |
| `food_order` | 9 | 0 | 0 | 100% | 100% | 100% |
| `ride` | 9 | 0 | 0 | 100% | 100% | 100% |
| `travel` | 3 | 0 | 0 | 100% | 100% | 100% |
| `shopping` | 0 | 0 | 0 | — | — | — |
| `music` | 2 | 0 | 0 | 100% | 100% | 100% |
| `video` | 4 | 0 | 0 | 100% | 100% | 100% |
| `tickets` | 1 | 0 | 0 | 100% | 100% | 100% |
| `reservation` | 2 | 0 | 0 | 100% | 100% | 100% |
| `task` | 0 | 0 | 0 | — | — | — |
| `note` | 0 | 0 | 0 | — | — | — |
| `bills` | 4 | 1 | 0 | 80% | 100% | 89% |
| `health` | 2 | 0 | 0 | 100% | 100% | 100% |
| `weather` | 2 | 0 | 1 | 100% | 67% | 80% |

## v4 work list (weakest categories first)

- **`edge_negation_marker`** — 0/1 passed. Add ~50 targeted training examples + reweight loss for these patterns.
- **`multi_intent`** — 3/6 passed. Add ~90 targeted training examples + reweight loss for these patterns.
- **`weather_minimal`** — 1/2 passed. Add ~50 targeted training examples + reweight loss for these patterns.
- **`smoke`** — 10/14 passed. Add ~120 targeted training examples + reweight loss for these patterns.
- **`negation`** — 6/8 passed. Add ~60 targeted training examples + reweight loss for these patterns.
- **`question_form`** — 4/5 passed. Add ~50 targeted training examples + reweight loss for these patterns.
