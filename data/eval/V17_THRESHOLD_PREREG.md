# v17 operating threshold - fixed BEFORE any real-traffic replay of v17

Written 2026-09-17, after the batteries and before either dogfood log was replayed through v17.
(A replay at the notebook's thresholds was started, then stopped and its log set aside
unread, so the blind 08-11 holdout has not informed this choice.)

**Thresholds: money 0.985, ride 0.985** (`PAYCHAT_CONV_THRESHOLDS=money=0.985,ride=0.985`)

## Why not the notebook's pick (money 0.988, ride 0.990)

The notebook takes the TOP of the F-beta plateau. That assumes positives saturate near
0.997, as v5 and v14 did. v17's positives stop at a lower ceiling, and the pick landed on it.
From `conv_model_v17/val_curve.json` (synthetic validation windows only):

```
ride    t=0.990  recall 0.8373   t=0.991  recall 0.0409   <- cliff 0.001 above the pick
money   t=0.994  recall 0.7476   t=0.995  recall 0.0702
precision is flat across 0.95-0.99 (money 0.886-0.892, ride 0.841-0.855)
```

On the batteries this showed as clear acceptances scoring 0.990 and not firing
("can you book me a cab" / "sure"): whether a real yes fires was being decided in the 4th
decimal place.

## Why 0.985

- inside the validation plateau (within 1% of peak F-beta) for BOTH labels
- 0.005 (ride) and 0.010 (money) below the cliff, where the curve is flat
- it is the second operating point the notebook printed before any result existed
  (suite at 0.985: must fire 95.1%, quiet 100%, held-out 88.6%)
- it is v14's production threshold, so v14 vs v17 compares like with like

## Notebook follow-up

The plateau-top rule must stay a safe margin below the recall cliff; fix before v18.

**Outcome (2026-09-23).** 0.985/0.985 shipped in `conv_model/model_info.json` and has been live
since 2026-09-20. The cliff-aware rule is now in `data_gen/make_v17b_notebook.py` and
`data_gen/make_ab_notebook.py`, and it reproduces this choice on v17's curve by itself: on the
validation slice it picks ride 0.985 (down from the plateau top 0.990) and leaves money at
0.988. An independent check while rebuilding the probe picked 0.985/0.985 for the shipped
model from its own validation data.
