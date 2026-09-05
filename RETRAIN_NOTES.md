# Retrain notes — read before training another conversation model

Seven conversation models have been trained since v5. **None of them shipped.** v5 is
still what production runs.

Two defects in the training pipeline were found on 2026-09-02, both measured, neither
previously known. They do not explain every failure, and one hypothesis that looked
certain turned out to be wrong. All three are written down here so the next round starts
from what was measured rather than from what was assumed.

---

## 1. The decision head is cold-started every round

The warm start copies the backbone and **not** the two tensors that decide when to fire.

```
                encoder    fire_head
v5 -> v6          3.8%       142.2%
v5 -> v7          3.8%       143.7%
v5 -> v8          4.1%       142.6%
v5 -> v9          4.2%       142.5%
v5 -> v10         6.5%       141.4%
v5 -> v11         6.1%       134.7%
v5 -> v12         3.7%       142.9%
```

`fire_head` is 1,536 parameters, 768x2. A cosine similarity of ~0 against v5's head means
it is not a drifted version of v5's — it is unrelated, which only happens from a random
start.

So every "retrain" has been *train a new classifier on a v5-ish backbone*. Nothing has
accumulated since v5 because nothing was ever built on it.

**Fixed by:** loading `fire_head.*` from the warm start alongside the encoder. Verified
in the three-arm run — arm B reached cosine 0.9978 to v5's head, arm C reached 0.99998.
The mechanism works and costs nothing.

**But it does not fix calibration.** See below.

---

## 2. The threshold rule has been backwards since v8

`model_info.json` from v8 onward records:

    threshold_selection: plateau centre, within 1% of peak F-beta(0.7)

F-beta with **beta < 1 weights precision above recall**. Taking the *plateau centre*
then walks the threshold DOWN from the peak — toward recall — which is the opposite of
what beta 0.7 asks for. The v11 notebook says so in its own words: *"a threshold at the
top of a plateau is a threshold chosen by noise"*, and picks the centre anyway.

```
v1..v7     thresholds 0.81 .. 0.985     no threshold_selection recorded
v8..v12    thresholds 0.52 .. 0.68      plateau centre, F-beta(0.7)
```

The collapse begins exactly where the rule was introduced. `data_gen/pick_thresholds.py`
implements the rule the notebook describes — highest threshold within tolerance of peak,
not the centre.

**This is a real defect and worth fixing.** It is NOT why v12 failed. See below.

---

## 3. What was disproven — read this before believing either of the above

Both defects are real. **Neither explains the failures**, and assuming they did would
send the next round in the wrong direction.

The three-arm run, same data and same warm start, changing only inheritance and LR:

```
arm            inherit  lr      epochs   val_f1   head cosine   thresholds
A_control        no     3e-5      4      0.8995     0.0016      0.580 / 0.545
B_inherited     yes     3e-5      4      0.8988     0.9978      0.560 / 0.545
C_gentle        yes     5e-6      1      0.8949     0.99998     0.615 / 0.565
```

Arm C's head is 99.998% identical to v5's and it still calibrated to 0.615, not 0.97.
**Head inheritance does not restore v5's calibration.**

And v12 on Switchboard — 7,432 utterances of real conversation containing no money and
no rides:

```
v5  (production)                    4 fires    0.54 per 1,000
v12 at its own 0.535 / 0.580        7          0.94
v12 at v5's   0.970 / 0.825         6          0.81
```

Forcing v12 to v5's strictest thresholds removed one false fire out of seven. The
remaining six fire at 0.97+, so v12 is *confidently* wrong on ordinary conversation, not
borderline. **The threshold is not why v12 fails either.**

What v12 actually did: it learned the offer/commitment distinction (basics 75 -> 80/82)
and became more willing to fire, and those are the same change. They do not separate at
any operating point.

---

## What to do differently next time

1. **Inherit the head.** Free, verified, and every round since v5 has thrown it away.
2. **Use `data_gen/pick_thresholds.py`.** The notebook's rule contradicts its own comment.
3. **Do not trust an explanation you have not tested.** Two confident diagnoses died in
   one evening, both to a measurement that took minutes. The dilution story fit six
   models and did not fit v12; the head story fit all seven and did not survive arm C.
4. **Score on Switchboard before shipping.** It is the only check whose data nobody here
   wrote, and it is what caught v12. Every internal battery said v12 was better.
