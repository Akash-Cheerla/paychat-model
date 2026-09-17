"""Build colab_conv_v17b/train_conv_v17b_colab.ipynb from v16's notebook.

Same model, seed, LR, epochs, batch, loss and warm start as v16 and v17: v17b changes the
DATA (data_gen/build_conv_v17b.py) and two things in how the run is judged:

  thresholds  v17's plateau-top pick sat 0.001 under ride's recall cliff. The pick now steps
              down until recall does not collapse within +0.005 (reproduces 0.985 on v17).
  self-check  the embedded suite is 835 cases scored on the EXACT label set, with a per-family
              "no family loses to the best earlier model" rule and the 21 conversations v17 lost.

    python data_gen/make_v17b_notebook.py
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
OUTDIR = ROOT / "colab_conv_v17b"
OUT = OUTDIR / "train_conv_v17b_colab.ipynb"
ZIP = ROOT / "conv_windows_v17b.zip"
WARM = ROOT / "warmstart_conv_v5.zip"

# Measured 2026-09-17 with tests/suite_scores_model.py on the 835-case suite, exact label set.
# v14 and v16 at their own thresholds; v17 at 0.985, the threshold it was evaluated at.
BASE = {
    "v14": {"fire": 66.3, "quiet": 96.4, "train": 76.2, "heldout": 73.1, "original": 73.4, "added": 80.5},
    "v16": {"fire": 72.4, "quiet": 95.6, "train": 81.5, "heldout": 73.9, "original": 78.0, "added": 83.2},
    "v17": {"fire": 92.8, "quiet": 96.0, "train": 96.7, "heldout": 86.3, "original": 96.6, "added": 86.3},
}
# The families added for v17b, all phrasings: (correct, total) for v14 and v17.
FAMILY_BASE = {
    "N10 offer alone":                                        {"v14": (8, 16),  "v17": (16, 16)},
    "N11 debt, reply only admits it (ruling 8)":              {"v14": (25, 25), "v17": (25, 25)},
    "N12 shortfall, polite noise":                            {"v14": (15, 16), "v17": (15, 16)},
    "N13 3rd person's bare ok after a clear take (ruling 6)": {"v14": (14, 14), "v17": (5, 14)},
    "P10 debt, reply agrees to pay (ruling 8)":               {"v14": (19, 27), "v17": (20, 27)},
    "P11 shortfall, then a commitment":                       {"v14": (12, 12), "v17": (12, 12)},
    "P12 clear take answers a request (s3c)":                 {"v14": (13, 14), "v17": (11, 14)},
    "P13 3rd person's clear words after a clear take":        {"v14": (10, 12), "v17": (11, 12)},
    "P14 2nd bare answer after a bare answer (ruling 7)":     {"v14": (0, 12),  "v17": (12, 12)},
    "P15 two open requests (s6a)":                            {"v14": (20, 28), "v17": (24, 28)},
    "P9 self-initiated send or book":                         {"v14": (46, 50), "v17": (44, 50)},
}

INTRO = """# PayChat - conversation classifier v17b (Colab)

**Same model and settings as v16/v17. Only the data and the checks changed.**

## What v17 showed (data/eval/V17_RESULTS.md)

v17 beat v14 on BOTH real logs - dev 09-10 caught 72.0 -> 78.0%, wrong prompts 12.3 -> 9.9%;
blind 08-11 caught 82.5 -> 87.5%, wrong 15.6 -> 14.6% - but lost behaviours the batteries cover,
all traced to its training batch:

```
"i'll send it" (self-initiated)                   0.997 -> 0.972   no future/offer contrast
"you owe me 20$" / "sure"                          0.997 -> 0.271   no debt wording at all
"can someone book me a cab" / "let me book a cab"  0.996 -> 0.218   read as an unprompted offer
   ... and then a 3rd person's "okay"             0.041 -> 0.990   A8 taught it (ruling 6)
```

Its notebook threshold also sat on a cliff: ride recall 0.837 at 0.990, 0.041 at 0.991.

## What v17b changes

- base relabel (`rulings_0917_relabel.py`): 41/10/8 windows like "you still owe me 200" /
  "my bad, sending now" were labelled quiet (FIRING_RULE s1a, ruling 8)
- A8 second answerer: the first answer is a bare sure/ok (ruling 7)
- 11 new families: self-initiated send/book, future alone, debt -> pay / admit, shortfall ->
  polite / commitment, clear take answering a request, 3rd person bare ok (quiet) / clear words
  (fire) after a clear take, two open requests
- thresholds: the plateau top, stepped down until recall holds within +0.005

**Data: `conv_windows_v17b.zip`** - 27,640 new windows (17,990 fire, 9,650 quiet) on
conv_windows_v16.zip + the 09-15 and 09-17 relabels. Every phrase is from the TRAIN half of
tests/robustness_lexicon.py; a guard fails the build if ANY message of a new window matches a
held-out phrasing, templates included.

## Ship criteria - v17b replaces v14 only if ALL hold

```
1  section 9b below         held-out > v17 86.3%, quiet >= 95.8%, original cases >= 95.6%,
                            added cases > v17 86.3%, NO added family below the best earlier
                            model by more than one case, all 21 regression conversations pass
2  tests/run_all.py         >= v14 per battery, nothing loses
3  82-case set              >= 75/82 (v14)
4  dev log 09-10            caught >= 72.0%, wrong prompts <= 12.3%  (and v17b should not lose to v17)
5  blind holdout 08-11      caught >= 82.5%, wrong prompts <= 15.6%
   4+5: strictly better than v14 on at least one of the four numbers
```

**Run every cell in order. Runtime -> Change runtime type -> GPU (A100 or L4).**
"""

SUITE_MD = """## 9b. Before you download anything: rule, regressions, families

835 robustness cases scored on the EXACT label set (a ride prompt on top of the right money
prompt is wrong), the 11 families added for v17b against the best earlier model, and the 21
conversations v17 lost. Every number below was measured locally on 2026-09-17.

If anything prints FAIL, do not download: send the output. The fix is data, not a re-run.
"""

SUITE_CODE = '''# Robustness self-check. Cases are embedded (no repo access needed).
import base64, gzip, json as _json
from collections import defaultdict

CASES = _json.loads(gzip.decompress(base64.b64decode(SUITE_B64)).decode("utf-8"))
REGRESS = _json.loads(gzip.decompress(base64.b64decode(REGRESS_B64)).decode("utf-8"))
BASE = __BASE__
FAMILY_BASE = __FAMILY_BASE__

def _probs(texts):
    out = []
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tokenizer(texts[i:i+64], return_tensors='pt', truncation=True,
                          max_length=MAX_LEN, padding=True).to(dev)
            out.extend(torch.sigmoid(model(b['input_ids'], b['attention_mask'])).float().cpu().tolist())
    return out

def _render(msgs):
    return render([{"s": s, "t": t} for s, t in msgs][-10:])

def suite_report(thresholds):
    fired_of = lambda p: [lab for lab, pr in zip(LABELS, p) if pr >= thresholds[lab]]
    fam = defaultdict(lambda: {'train': [0, 0], 'heldout': [0, 0]})
    tot = {k: [0, 0] for k in ('pos', 'neg', 'train', 'heldout', 'original', 'added')}
    for c, p in zip(CASES, _probs([_render(c['msgs']) for c in CASES])):
        ok = fired_of(p) == ([c['want']] if c['want'] else [])
        fam[c['family']][c['split']][0] += ok; fam[c['family']][c['split']][1] += 1
        for k in ('pos' if c['want'] else 'neg', c['split'], 'added' if c['family'] in FAMILY_BASE else 'original'):
            tot[k][0] += ok; tot[k][1] += 1
    print(f"{'family':56}{'train':>14}{'held-out':>14}")
    for k in sorted(fam):
        v = fam[k]
        cell = lambda s: f"{v[s][0]}/{v[s][1]} {100*v[s][0]/v[s][1]:5.1f}%" if v[s][1] else '-'
        print(f"{k:56}{cell('train'):>14}{cell('heldout'):>14}")
    got = {'fire': tot['pos'], 'quiet': tot['neg'], 'train': tot['train'], 'heldout': tot['heldout'],
           'original': tot['original'], 'added': tot['added']}
    got = {k: 100 * v[0] / max(v[1], 1) for k, v in got.items()}
    ok_all = True
    print(f"\\n{'criterion':24}{'this run':>9}{'v14':>7}{'v16':>7}{'v17':>7}   verdict")
    rules = [('HELD-OUT phrasings', 'heldout', lambda g: g > BASE['v17']['heldout']),
             ('must stay quiet', 'quiet', lambda g: g >= max(BASE['v14']['quiet'], BASE['v17']['quiet']) - 0.6),
             ('original cases', 'original', lambda g: g >= BASE['v17']['original'] - 1.0),
             ('added cases', 'added', lambda g: g > BASE['v17']['added']),
             ('must fire', 'fire', lambda g: g > BASE['v14']['fire'])]
    for label, key, rule in rules:
        good = rule(round(got[key], 1))
        ok_all &= good
        print(f"{label:24}{got[key]:>8.1f}%{BASE['v14'][key]:>6.1f}%{BASE['v16'][key]:>6.1f}%{BASE['v17'][key]:>6.1f}%   {'PASS' if good else 'FAIL'}")
    print(f"\\n{'added family (all phrasings)':56}{'this run':>10}{'v14':>8}{'v17':>8}   verdict (>= best - 1 case)")
    for k, b in FAMILY_BASE.items():
        v = fam[k]
        ok_n, n = v['train'][0] + v['heldout'][0], v['train'][1] + v['heldout'][1]
        best = max(b['v14'][0], b['v17'][0])
        good = ok_n >= best - 1
        ok_all &= good
        print(f"{k:56}{ok_n:>5}/{n:<4}{b['v14'][0]:>5}/{b['v14'][1]:<3}{b['v17'][0]:>4}/{b['v17'][1]:<3}   {'PASS' if good else 'FAIL'}")
    print(f"\\n{'regression conversation':40}{'want':>6}  {'money':>6} {'ride':>6}   verdict")
    for (name, want, turns), p in zip(REGRESS, _probs([_render(t) for _, _, t in REGRESS])):
        good = fired_of(p) == ([want] if want else [])
        ok_all &= good
        print(f"{name:40}{str(want):>6}  {p[0]:>6.3f} {p[1]:>6.3f}   {'PASS' if good else 'FAIL'}")
    print('\\n' + ('LOOKS GOOD - download, then run the batteries, the 82-case set and tests/ship_gate.py locally'
                  if ok_all else 'DO NOT DOWNLOAD - send this output; the data needs fixing, not another run'))
    return got

print('--- at the thresholds this run picked ---')
_ = suite_report({name: TH[i] for i, name in enumerate(LABELS)})
print('\\n--- at 0.985 for both (v14 production, v17 as evaluated) ---')
_ = suite_report({name: 0.985 for name in LABELS})
'''

THRESH_OLD = "    pick = best_hi\n"
THRESH_NEW = """    pick = best_hi
    # v17 (2026-09-17): the plateau top sat 0.001 under ride's recall CLIFF - recall 0.837 at
    # 0.990, 0.041 at 0.991 - so clear yeses scoring 0.990 did not fire. Step down until recall
    # holds (>= 90% of its value) across the next +0.005. On v17's curve this gives ride 0.985
    # and leaves money at 0.988; on a curve with no cliff near the top it changes nothing.
    MARGIN = 0.005
    _rs = np.array(rs)
    while pick > best_lo:
        _ahead = [k for k in range(pick, len(ts)) if ts[k] <= ts[pick] + MARGIN + 1e-9]
        if _rs[_ahead].min() >= 0.9 * _rs[pick]:
            break
        pick -= 1
    if pick != best_hi:
        print(f'         {name}: plateau top {ts[best_hi]:.3f} is within {MARGIN} of a recall cliff; using {ts[pick]:.3f}')
"""


def gz(obj):
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode("ascii")   # mtime=0: same bytes every build


def suite_cases():
    import robustness_lexicon as L
    import robustness_suite as S
    cases = []
    for c in S.build():
        if c["family"].startswith("S"):
            continue
        cases.append({"family": c["family"], "want": c["want"], "msgs": c["msgs"],
                      "split": "heldout" if any(L.is_heldout(p) for p in c["tested"]) else "train"})
    return cases


def regress_cases():
    """The conversations v17 lost (batteries, 82-case set) and the 09-17 ruling examples.
    Test phrasings: the build's held-out guard keeps their templates out of training."""
    PAD = None  # noqa: F841 (kept aligned with scratchpad/regress_scores.py)
    return [
        ["self: sending now", "money", [["1", "sending now"]]],
        ["self: i'll send it", "money", [["1", "i'll send it"]]],
        ["self: sending dollar I owe", "money", [["1", "I'll be sending you that dollar that I owe you"]]],
        ["self: Sure i will send it now", "money", [["1", "Sure i will send it now"]]],
        ["self: im sending you 20 now", "money", [["1", "im sending you 20 now"]]],
        ["self: booking now", "ride", [["1", "booking now"]]],
        ["self: Okay I'll get you a cab", "ride", [["1", "Okay I'll get you a cab for it then"]]],
        ["offer: let me send you 500", None, [["1", "let me send you 500"]]],
        ["future: i'll pay on the 1st", None, [["1", "i'll pay on the 1st"]]],
        ["owe: you owe me 20$ / sure", "money", [["1", "you owe me 20$"], ["2", "sure"]]],
        ["owe: you owe me 500 rupees / sure", "money", [["1", "you owe me 500 rupees"], ["2", "sure"]]],
        ["owe: still owe 300 / alright", None, [["1", "you still owe me 300"], ["2", "alright"]]],
        ["owe: still owe 300 / will send now", "money", [["1", "you still owe me 300"], ["2", "oh i forgot, will send now"]]],
        ["short: im short 300 / sure", None, [["1", "im short 300 this month"], ["2", "sure"]]],
        ["short: im short 300 / will send", "money", [["1", "im short 300 this month"], ["2", "im sorry, will send"]]],
        ["g09: B let me book / C okay", None, [["A", "can someone book me a cab to HSR"], ["B", "let me book a cab"], ["C", "okay"]]],
        ["g09b: B let me book (fires)", "ride", [["A", "can someone book me a cab to HSR"], ["B", "let me book a cab"]]],
        ["r3: B sure / C sure", "money", [["A", "can someone send me 40 for lunch"], ["B", "sure"], ["C", "sure"]]],
        ["g04: pizza, cab / ok sending", "money", [["A", "venmo me 20 for pizza"], ["C", "also book me a cab"], ["B", "ok sending"]]],
        ["g06: pizza, cab / ok", "ride", [["A", "venmo me 20 for pizza"], ["C", "also book me a cab"], ["B", "ok"]]],
        ["other one", "money", [["20", "Can you send me 60 for the movie tickets"], ["20", "Also send me 40 for lunch"],
                                ["4", "Sending the 40 now"], ["P", "lol"], ["Q", "ok"], ["P", "nice"], ["Q", "haha"],
                                ["P", "k"], ["4", "And sending the other one too"]]],
    ]


def main():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        s = "".join(c["source"])
        s = s.replace("conv_windows_v16", "conv_windows_v17b").replace("conv_model_v16", "conv_model_v17b")
        s = s.replace("train_conv_v16", "train_conv_v17b")
        s = s.replace("print('all four ship criteria pass. Then:')",
                      "print('every ship criterion in the first cell passes. Then locally:')")
        s = s.replace("print('  bash tests/run_v26_gate.sh ./saved_model ./conv_model_v17b /tmp/scratch')",
                      "print('  python tests/ship_gate.py --url <server running conv_model_v17b> --name v17b')")
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        c["source"] = s.splitlines(keepends=True)
    nb["cells"][0]["source"] = INTRO.splitlines(keepends=True)

    # stale-data guard + fingerprint of this exact zip (see make_v17_notebook.py for why)
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
    check = ("\nEXPECT = " + json.dumps(expect) + "   # conv_windows_v17b.zip as built locally\n"
             "import hashlib\n"
             "_got = {s: f\"{len(v)}:\" + hashlib.sha1('\\n'.join(sorted(r['conv'] for r in v)).encode()).hexdigest()[:12]\n"
             "        for s, v in DATA.items()}\n"
             "assert _got == EXPECT, (f'WRONG DATA: {_got} != {EXPECT}. Runtime > Disconnect and delete '\n"
             "                        'runtime, then upload the current conv_windows_v17b.zip')\n"
             "print('data matches the local build:', EXPECT)\n")
    anchor = "    print(f'{s:<6}{len(v):>7} windows | {f:>5} fire ({f/len(v)*100:.1f}%)')\n"
    assert s.count("if not os.path.exists('data/train.json'):") == 1 and s.count(anchor) == 1
    s = s.replace("if not os.path.exists('data/train.json'):", guard, 1).replace(anchor, anchor + check, 1)
    load["source"] = s.splitlines(keepends=True)

    # cliff-aware threshold pick
    th = next(c for c in nb["cells"] if c["cell_type"] == "code" and "PLATEAU = " in "".join(c["source"]))
    s = "".join(th["source"])
    assert s.count(THRESH_OLD) == 1, "threshold cell changed shape"
    th["source"] = s.replace(THRESH_OLD, THRESH_NEW, 1).splitlines(keepends=True)

    cases = suite_cases()
    fams = {c["family"] for c in cases}
    assert set(FAMILY_BASE) <= fams, f"FAMILY_BASE names missing from the suite: {set(FAMILY_BASE) - fams}"
    code = SUITE_CODE.replace("__BASE__", json.dumps(BASE)).replace("__FAMILY_BASE__", json.dumps(FAMILY_BASE))
    at = next(i for i, c in enumerate(nb["cells"])
              if c["cell_type"] == "markdown" and "## 10." in "".join(c["source"]))
    nb["cells"][at:at] = [
        {"cell_type": "markdown", "metadata": {}, "source": SUITE_MD.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": [f'SUITE_B64 = "{gz(cases)}"   # {len(cases)} robustness cases\n',
                    f'REGRESS_B64 = "{gz(regress_cases())}"   # {len(regress_cases())} regression conversations\n']},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": code.splitlines(keepends=True)},
    ]
    left = [i for i, c in enumerate(nb["cells"]) if i > 0
            and any(f in "".join(c["source"]) for f in ("conv_windows_v16", "conv_model_v16", "train_conv_v16"))]
    assert not left, f"cells still reference v16 files: {left}"
    assert not any("run_v26_gate" in "".join(c["source"]) for c in nb["cells"]), "stale gate instruction left"
    OUTDIR.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    for src in (ZIP, WARM):
        dst = OUTDIR / src.name
        if not (dst.exists() and filecmp.cmp(src, dst, shallow=False)):
            shutil.copy(src, dst)
            print(f"  copied {src.name}")
        assert filecmp.cmp(src, dst, shallow=False), f"{dst} differs from {src}"
    print(f"  wrote {OUT.relative_to(ROOT)}  ({len(cases)} suite cases, {len(regress_cases())} regression conversations)")
    print(f"  upload to Colab: {ZIP.name} and {WARM.name} (both in {OUTDIR.name}/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
