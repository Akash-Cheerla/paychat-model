# FYOE v3 — seed baseline report

**Generated:** 2026-05-13 18:12  
**Model:** roberta-base v3 (125M params, device: `cpu`)  
**IID test-set accuracy (training distribution):** 98.72%  
**Seed-suite size:** 127 curated real-world cases  
**Latency:** 156 ms / case (single-batch CPU)

## TL;DR

- **127/127 passed (100.0%)** on the seed suite
- **0 failures** — these become v4 training data
- Reality gap: **-1.3 percentage points** between IID test set and adversarial seed suite
- This gap is the work item. Closing it = production-ready model.

## Performance by category

| Tag | Cases | Passed | Pass rate | Status |
|---|---:|---:|---:|---|
| `smoke` | 14 | 14 | 100% | PERFECT |
| `multi_intent` | 6 | 6 | 100% | PERFECT |
| `numerics` | 3 | 3 | 100% | PERFECT |
| `slang` | 4 | 4 | 100% | PERFECT |
| `typo` | 1 | 1 | 100% | PERFECT |
| `long` | 1 | 1 | 100% | PERFECT |
| `negation` | 8 | 8 | 100% | PERFECT |
| `past_tense` | 7 | 7 | 100% | PERFECT |
| `ambiguous` | 4 | 4 | 100% | PERFECT |
| `chitchat` | 6 | 6 | 100% | PERFECT |
| `false_positive_bait` | 7 | 7 | 100% | PERFECT |
| `code_mixed` | 4 | 4 | 100% | PERFECT |
| `edge_pronoun` | 3 | 3 | 100% | PERFECT |
| `edge_negation_marker` | 1 | 1 | 100% | PERFECT |
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
| `weather_minimal` | 2 | 2 | 100% | PERFECT |
| `v5_context_buildup` | 3 | 3 | 100% | PERFECT |
| `v5_confirmation` | 3 | 3 | 100% | PERFECT |
| `v5_confirm_negative` | 2 | 2 | 100% | PERFECT |
| `v5_context_negative` | 2 | 2 | 100% | PERFECT |
| `v5_correction` | 1 | 1 | 100% | PERFECT |
| `v5_no_carryforward` | 1 | 1 | 100% | PERFECT |
| `v5_disambiguation` | 2 | 2 | 100% | PERFECT |
| `v5_single_with_context` | 1 | 1 | 100% | PERFECT |
| `question_form` | 5 | 5 | 100% | PERFECT |
| `idiomatic_false_pos` | 8 | 8 | 100% | PERFECT |
| `present_activity` | 3 | 3 | 100% | PERFECT |
| `third_person` | 3 | 3 | 100% | PERFECT |
| `ultra_short` | 3 | 3 | 100% | PERFECT |
| `bare_word` | 5 | 5 | 100% | PERFECT |

## All 0 failures (grouped by category)

## Per-intent precision / recall (on seed suite)

| Intent | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `money` | 17 | 0 | 0 | 100% | 100% | 100% |
| `alarm` | 9 | 0 | 0 | 100% | 100% | 100% |
| `contact` | 6 | 0 | 0 | 100% | 100% | 100% |
| `calendar` | 4 | 0 | 0 | 100% | 100% | 100% |
| `maps` | 6 | 0 | 0 | 100% | 100% | 100% |
| `food_order` | 10 | 0 | 0 | 100% | 100% | 100% |
| `ride` | 10 | 0 | 0 | 100% | 100% | 100% |
| `travel` | 3 | 0 | 0 | 100% | 100% | 100% |
| `shopping` | 0 | 0 | 0 | — | — | — |
| `music` | 3 | 0 | 0 | 100% | 100% | 100% |
| `video` | 4 | 0 | 0 | 100% | 100% | 100% |
| `tickets` | 1 | 0 | 0 | 100% | 100% | 100% |
| `reservation` | 2 | 0 | 0 | 100% | 100% | 100% |
| `task` | 0 | 0 | 0 | — | — | — |
| `note` | 0 | 0 | 0 | — | — | — |
| `bills` | 4 | 0 | 0 | 100% | 100% | 100% |
| `health` | 2 | 0 | 0 | 100% | 100% | 100% |
| `weather` | 4 | 0 | 0 | 100% | 100% | 100% |

## v4 work list (weakest categories first)

