"""Build colab_conv_v17/train_conv_v17_colab.ipynb from v16's notebook.

v17 changes the DATA only: v16's windows with the 2026-09-15 relabel, plus 14,830 coverage
windows (data_gen/build_conv_v17.py). Seed, LR, epochs, batch, loss, warm start and the
threshold grid are v16's, so a v16 -> v17 difference can only be the data.

    python data_gen/make_v17_notebook.py
"""
import base64
import filecmp
import gzip
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
SRC = ROOT / "colab_conv_v16/train_conv_v16_colab.ipynb"
OUTDIR = ROOT / "colab_conv_v17"
OUT = OUTDIR / "train_conv_v17_colab.ipynb"
ZIP = ROOT / "conv_windows_v17.zip"
WARM = ROOT / "warmstart_conv_v5.zip"

INTRO = """# PayChat - conversation classifier v17 (Colab)

**One change from v16: the data.** Same seed, same warm start (v5), same hyperparameters,
same threshold grid. Anything that moves is the data.

**Data: `conv_windows_v17.zip`** = `conv_windows_v16.zip`
+ 200 train / 18 val / 21 test labels corrected to Akash's 2026-09-15 ruling 1
+ **22,140 coverage windows** (21,033 train, 1,107 val; 14,140 fire, 8,000 quiet).
Train is now 338,042 windows, 14.5% fire.

---

## Why this batch

v6 to v16 were a random walk because each round added more of the same phrasings. Probing
v14 on its real-traffic misses showed the failure is not judgement, it is vocabulary - one
word flips it:

```
"can you give me 10₹"  / "ok"    0.08        "can you send me $10" / "ok"    0.997
"...spot 200 till then?" / "yep fr" 0.03     same request          / "yep"   0.997
"can you book a cab..."  / "why not" 0.03    same request          / "sure"  0.996
"shall i send you the money now?" / "yes please"  0.04
```

Its training data had **zero** fire windows ending in `why not` (102 quiet), `oh ok`
(32 quiet), and **zero** windows containing an amount written `10₹`.

## What is in the batch

| family | windows | teaches |
|---|---|---|
| A1/A2 acceptance wording | 6,600 | many ways to say yes to a request |
| A3/A4 request wording, amount formats | 2,000 | `give me` / `spot` / `wire`, `₹500` / `rs 500` / `500 rupees` |
| A5 agreement + detail question | 3,200 | ruling 1: `sure, how much?` fires |
| A6 offer -> acceptance | 720 | fires on the acceptance, not the offer |
| A7 interruption before the acceptance | 720 | a `hi` in between does not cancel it |
| A8 second group answerer | 600 | ruling 3 |
| A9 re-sent after a failure | 300 | ruling 5 |
| B1-B12 quiet | 8,000 | declines, questions, future, already done, hedges, looks-like-yes, the requester's own follow-up, acceptance words with no request, offer alone, offer declined, re-sent request (ruling 2), answer after `sent it` (ruling 3), request alone |

Every phrase comes from `tests/robustness_lexicon.py`, which is **split in half**: only the
TRAIN half is here. The HELDOUT half - including every phrasing seen in dogfood - is
reserved for `tests/robustness_suite.py`, so the suite measures whether v17 learned the
rule or the words. No real chat text is used as training data.

## The gate passes

`python data_gen/gate_conv_v12.py conv_windows_v16.zip conv_windows_v17.zip` - no role
loses fire rate. Rises are the batch working: ack_money 73.6 -> 78.5, ack_ride 73.2 ->
80.2, ack_after_offer 77.7 -> 81.1, question 0.0 -> 23.4 (ruling 1).

## Ship criteria - v17 replaces v14 only if ALL hold

Bars are v14 measured 2026-09-16 with today's server and the rulings-revised labels.

```
1  section 9b below (in this notebook)  held-out phrasings > 69.6% (v16, the best so far)
                                        must-stay-quiet ~100%
2  tests/run_all.py batteries           >= v14 per battery, nothing loses
3  82-case context/group set            >= 75/82   (v14)
4  real traffic, 09-10 dev log          caught >= 72.0% and wrong prompts <= 12.3%   (timing-aware)
5  blind holdout 08-11                  caught >= 82.5% and wrong prompts <= 15.6%
   4+5 together: strictly better than v14 on at least one of those four numbers
6  already done stays quiet, self-initiated still fires
   (tests/test_basics.py 'already done', tests/test_self_initiated.py)
```

Criteria 4 and 5 are the ones v6-v16 never had. A model that wins the suite but not real
traffic does not ship.

**Run every cell in order. Runtime -> Change runtime type -> GPU (A100 or L4).**
"""


SUITE_MD = """## 9b. Does it know the RULE or just the words? (before you download anything)

The robustness suite, scored against the classifier alone. Every phrase list was split in
half: the TRAIN half is in the training data, the HELD-OUT half is not, and every phrasing
seen in real dogfood is held out. **Held-out is the number that matters** - it is the one
v6-v16 never measured.

Scored on exactly these cases (2026-09-16, classifier only, each model's own thresholds):

```
                     must fire   must stay quiet   train phrasings   HELD-OUT
v14  (in git)          62.2%         100.0%            76.5%           64.6%
v15  (local)           66.9%         100.0%            79.8%           67.7%
v16  (local)           68.8%         100.0%            80.9%           69.6%
```

**v17 must beat v16's held-out 69.6%, not just v14's.** A paired CPU experiment before this
run (same model, top layers only, 1 epoch, 2 seeds, arms differing ONLY in the added
windows) moved held-out 67.6% -> 75.3% with a seed-to-seed spread under 1 point, so the
batch does carry the rule. What it did NOT move was `P6 agreement + question` (ruling 1),
0% -> 3.6%: that family fights a prior learned from ~3,000 base windows, and 3,200 new
windows over 4 full epochs is its real test. Watch that row.

If v17 is not clearly better on held-out phrasings while holding must-stay-quiet near 100%,
it does not ship, and the fix is more data variety - not another training run with the same
data.
"""

SUITE_CODE = '''# Robustness suite, classifier only. Cases are embedded (no repo access needed).
import base64, gzip, json as _json
from collections import defaultdict

CASES = _json.loads(gzip.decompress(base64.b64decode(SUITE_B64)).decode("utf-8"))
V14 = {"fire": 62.2, "quiet": 100.0, "train": 76.5, "heldout": 64.6}   # measured 2026-09-16
V16 = {"fire": 68.8, "quiet": 100.0, "train": 80.9, "heldout": 69.6}   # the strongest local model

def _render_case(c):
    win = [{"s": s, "t": t} for s, t in c["msgs"]][-10:]
    return render(win)

def suite_report(thresholds):
    texts = [_render_case(c) for c in CASES]
    probs = []
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tokenizer(texts[i:i+64], return_tensors='pt', truncation=True,
                          max_length=MAX_LEN, padding=True).to(dev)
            probs.extend(torch.sigmoid(model(b['input_ids'], b['attention_mask'])).float().cpu().tolist())
    fam = defaultdict(lambda: {'train': [0, 0], 'heldout': [0, 0]})
    tot = {'pos': [0, 0], 'neg': [0, 0], 'train': [0, 0], 'heldout': [0, 0]}
    for c, p in zip(CASES, probs):
        fired = next((lab for lab, pr in zip(LABELS, p) if pr >= thresholds[lab]), None)
        ok = (fired == c['want'])
        split = c['split']
        fam[c['family']][split][0] += ok; fam[c['family']][split][1] += 1
        side = 'pos' if c['want'] else 'neg'
        tot[side][0] += ok; tot[side][1] += 1
        tot[split][0] += ok; tot[split][1] += 1
    print(f"{'family':50}{'train':>16}{'held-out':>16}")
    for k in sorted(fam):
        v = fam[k]
        cell = lambda s: f"{v[s][0]}/{v[s][1]} {100*v[s][0]/v[s][1]:5.1f}%" if v[s][1] else '-'
        print(f"{k:50}{cell('train'):>16}{cell('heldout'):>16}")
    got = {k: 100 * tot[k][0] / max(tot[k][1], 1) for k in ('train', 'heldout')}
    got['fire'] = 100 * tot['pos'][0] / max(tot['pos'][1], 1)
    got['quiet'] = 100 * tot['neg'][0] / max(tot['neg'][1], 1)
    print(f"\\n{'criterion':28}{'v17':>8}{'v14':>8}{'v16':>8}   verdict (must beat v16)")
    rules = [('must fire', 'fire', '>'), ('must stay quiet', 'quiet', '>='),
             ('train phrasings', 'train', '>'), ('HELD-OUT phrasings', 'heldout', '>')]
    ok_all = True
    for label, key, op in rules:
        bar = max(V14[key], V16[key])
        # compare at the printed precision: unrounded, v16 itself (69.62) "beat" v16 (69.6)
        good = round(got[key], 1) > bar if op == '>' else round(got[key], 1) >= bar - 0.6
        ok_all &= good
        print(f"{label:28}{got[key]:>7.1f}%{V14[key]:>7.1f}%{V16[key]:>7.1f}%   {'PASS' if good else 'FAIL'}")
    print('\\n' + ('LOOKS GOOD - now run the batteries and the real logs locally (tests/ship_gate.py)'
                  if ok_all else 'DOES NOT BEAT v14 on the suite - do not ship; the data needs more variety'))
    return got

print('--- at the thresholds this run picked ---')
_ = suite_report({name: TH[i] for i, name in enumerate(LABELS)})
print('\\n--- at v14 thresholds (0.985), same numbers as the v14 line above ---')
_ = suite_report({name: 0.985 for name in LABELS})
'''


def suite_b64():
    """The suite's cases, minus the pipeline-only sequence families, as one gzipped blob."""
    import robustness_lexicon as L
    import robustness_suite as S
    cases = []
    for c in S.build():
        if c["family"].startswith("S"):
            continue
        cases.append({"family": c["family"], "want": c["want"], "msgs": c["msgs"],
                      "split": "heldout" if any(L.is_heldout(p) for p in c["tested"]) else "train"})
    raw = json.dumps(cases, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode("ascii"), len(cases)   # mtime=0: same bytes every build


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        s = "".join(c["source"])
        s = s.replace("conv_windows_v16", "conv_windows_v17").replace("conv_model_v16", "conv_model_v17")
        s = s.replace("train_conv_v16", "train_conv_v17")
        s = s.replace("print('all four ship criteria pass. Then:')",
                      "print('every ship criterion in the first cell passes. Then locally:')")
        s = s.replace("print('  bash tests/run_v26_gate.sh ./saved_model ./conv_model_v17 /tmp/scratch')",
                      "print('  python tests/ship_gate.py --url <server running conv_model_v17> --name v17')")
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        c["source"] = s.splitlines(keepends=True)
    nb["cells"][0]["source"] = INTRO.splitlines(keepends=True)

    # The load cell skipped extraction whenever data/train.json existed, so re-uploading a fixed
    # zip into the same runtime silently trained on the OLD data. Always re-extract, and refuse
    # to go on unless the counts are exactly this zip's.
    # Counts alone are not enough: the split fix kept every count identical. Fingerprint the
    # conversation ids in each split (same expression runs in the notebook).
    def fingerprint(rows):
        return f"{len(rows)}:" + hashlib.sha1("\n".join(sorted(r["conv"] for r in rows)).encode()).hexdigest()[:12]
    with zipfile.ZipFile(ZIP) as z:
        expect = {s: fingerprint(json.loads(z.read(f"{s}.json"))) for s in ("train", "val", "test")}
    load = next(c for c in nb["cells"] if c["cell_type"] == "code"
                and "if not os.path.exists('data/train.json'):" in "".join(c["source"]))
    s = "".join(load["source"])
    guard = ("# A zip uploaded now must replace data/ left by an earlier upload in this runtime.\n"
             "if os.path.exists('data'):\n"
             "    shutil.rmtree('data'); print('removed data/ from an earlier upload - re-extracting')\n\n"
             "if not os.path.exists('data/train.json'):")
    check = ("\nEXPECT = " + json.dumps(expect) + "   # conv_windows_v17.zip as built locally\n"
             "import hashlib\n"
             "_got = {s: f\"{len(v)}:\" + hashlib.sha1('\\n'.join(sorted(r['conv'] for r in v)).encode()).hexdigest()[:12]\n"
             "        for s, v in DATA.items()}\n"
             "assert _got == EXPECT, (f'WRONG DATA: {_got} != {EXPECT}. Runtime > Disconnect and delete '\n"
             "                        'runtime, then upload the current conv_windows_v17.zip')\n"
             "print('data matches the local build:', EXPECT)\n")
    anchor = "    print(f'{s:<6}{len(v):>7} windows | {f:>5} fire ({f/len(v)*100:.1f}%)')\n"
    assert s.count("if not os.path.exists('data/train.json'):") == 1 and s.count(anchor) == 1
    s = s.replace("if not os.path.exists('data/train.json'):", guard, 1).replace(anchor, anchor + check, 1)
    load["source"] = s.splitlines(keepends=True)

    # The suite self-check goes in right after the thresholds cell, so the run says whether
    # it beat v14 before anyone downloads the model.
    b64, n_cases = suite_b64()
    at = next(i for i, c in enumerate(nb["cells"])
              if c["cell_type"] == "markdown" and "## 10." in "".join(c["source"]))
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": SUITE_MD.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [f'SUITE_B64 = "{b64}"   # {n_cases} robustness cases\n']},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": SUITE_CODE.splitlines(keepends=True)},
    ]
    nb["cells"][at:at] = cells
    # cell 0 is the intro, which names conv_windows_v16.zip on purpose (it is the base).
    left = [i for i, c in enumerate(nb["cells"]) if i > 0
            and any(f in "".join(c["source"]) for f in ("conv_windows_v16", "conv_model_v16", "train_conv_v16"))]
    assert not left, f"cells still reference v16 files: {left}"
    assert not any("run_v26_gate" in "".join(c["source"]) for c in nb["cells"]), "stale gate instruction left"
    OUTDIR.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    # Byte comparison, not size: a rebuilt data zip can keep its exact size and still differ
    # (the split fix did), and a size check left the broken zip in the upload folder.
    for src in (ZIP, WARM):
        dst = OUTDIR / src.name
        if not (dst.exists() and filecmp.cmp(src, dst, shallow=False)):
            shutil.copy(src, dst)
            print(f"  copied {src.name}")
        assert filecmp.cmp(src, dst, shallow=False), f"{dst} differs from {src}"
    print(f"  wrote {OUT.relative_to(ROOT)}")
    print(f"  upload to Colab: {ZIP.name} and {WARM.name} (both in {OUTDIR.name}/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
