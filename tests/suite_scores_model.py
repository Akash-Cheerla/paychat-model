"""Score the robustness suite against the CLASSIFIER ALONE - no server, no pipeline.

The suite normally runs through /detect, which mixes the classifier with the server's
dedup rules. To compare a freshly trained model against v14 (and to run the same check
inside the Colab notebook), the model has to be scored on its own.

Windows are rendered exactly as conv_classifier.render does, so this is what the model
sees in production.

    python tests/suite_scores_model.py --onnx conv_model/conv_model.onnx --name v14
    python tests/suite_scores_model.py --torch conv_model --name v17
"""
import argparse, io, json, re, sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from conv_classifier import render          # noqa: E402
import robustness_lexicon as L              # noqa: E402
import robustness_suite as S                # noqa: E402

LABELS = ("money", "ride")


def windows(cases):
    """Suite case -> the window the classifier sees for its final message."""
    out = []
    for c in cases:
        if c["family"].startswith("S"):
            continue                        # sequence cases are pipeline behaviour
        win = [{"s": s, "t": t} for s, t in c["msgs"]][-10:]
        out.append((render(win), c))
    return out


def scores_onnx(path, texts, tok_dir, max_len):
    import numpy as np, onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    tok.truncation_side = "left"
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    out = []
    for i in range(0, len(texts), 32):
        b = tok(texts[i:i + 32], return_tensors="np", truncation=True, max_length=max_len, padding=True)
        feed = {k: v for k, v in b.items() if k in names}
        logits = sess.run(None, feed)[0]
        out.extend(1 / (1 + np.exp(-logits)))
    return out


def scores_torch(model_dir, texts, max_len):
    import torch
    from transformers import AutoTokenizer, RobertaModel, RobertaConfig
    from safetensors.torch import load_file
    import torch.nn as nn
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.truncation_side = "left"
    cfg = RobertaConfig.from_pretrained(model_dir)
    backbone = RobertaModel(cfg, add_pooling_layer=False)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.roberta = backbone
            self.fire_head = nn.Linear(cfg.hidden_size, len(LABELS))

        def forward(self, **b):
            return self.fire_head(self.roberta(**b).last_hidden_state[:, 0, :])

    m = M()
    m.load_state_dict(load_file(Path(model_dir) / "model.safetensors"), strict=False)
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            b = tok(texts[i:i + 16], return_tensors="pt", truncation=True, max_length=max_len, padding=True)
            out.extend(torch.sigmoid(m(**b)).tolist())
    return out


NEW_0917 = re.compile(r"^(P(9|1\d)|N1\d) ")     # families added for v17b


def report(cases_scored, name, thresholds):
    fam = defaultdict(lambda: {"train": [0, 0], "heldout": [0, 0]})
    tot = {"pos": [0, 0], "neg": [0, 0], "train": [0, 0], "heldout": [0, 0]}
    sub = {"original": [0, 0], "added 09-17": [0, 0]}
    for c, p in cases_scored:
        fired = [lab for lab, pr in zip(LABELS, p) if pr >= thresholds[lab]]
        # EXACT label set (2026-09-17). The first-label-wins check let a ride fire on top of the
        # right money prompt pass, which is exactly what v17 did on two open requests.
        ok = fired == ([c["want"]] if c["want"] else [])
        split = "heldout" if any(L.is_heldout(x) for x in c["tested"]) else "train"
        fam[c["family"]][split][0] += ok
        fam[c["family"]][split][1] += 1
        side = "pos" if c["want"] else "neg"
        tot[side][0] += ok; tot[side][1] += 1
        tot[split][0] += ok; tot[split][1] += 1
        s = sub["added 09-17" if NEW_0917.match(c["family"]) else "original"]
        s[0] += ok; s[1] += 1
    print(f"\n  classifier-only suite - {name} - {len(cases_scored)} cases   thresholds {thresholds}")
    print(f"  {'family':50}{'train':>16}{'held-out':>16}")
    for k in sorted(fam):
        v = fam[k]
        cell = lambda s: f"{v[s][0]}/{v[s][1]} {100*v[s][0]/v[s][1]:5.1f}%" if v[s][1] else "-"
        print(f"  {k:50}{cell('train'):>16}{cell('heldout'):>16}")
    pct = lambda a: f"{a[0]}/{a[1]} = {100*a[0]/max(a[1],1):.1f}%"
    print(f"\n  must fire, fired:        {pct(tot['pos'])}")
    print(f"  must stay quiet, quiet:  {pct(tot['neg'])}")
    print(f"  train phrasings:         {pct(tot['train'])}")
    print(f"  held-out phrasings:      {pct(tot['heldout'])}")
    for k, v in sub.items():
        print(f"  {k + ' cases:':25}{pct(v)}")
    out = {"fire": 100 * tot["pos"][0] / max(tot["pos"][1], 1), "quiet": 100 * tot["neg"][0] / max(tot["neg"][1], 1),
           "train": 100 * tot["train"][0] / max(tot["train"][1], 1), "heldout": 100 * tot["heldout"][0] / max(tot["heldout"][1], 1)}
    out.update({k: 100 * v[0] / max(v[1], 1) for k, v in sub.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--torch", dest="torch_dir", default=None)
    ap.add_argument("--tokenizer", default="conv_model")
    ap.add_argument("--name", default="model")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=None, help="one threshold for both labels")
    a = ap.parse_args()
    info = json.loads((ROOT / a.tokenizer / "model_info.json").read_text())
    th = {k: (a.threshold if a.threshold is not None else info["thresholds"][k]) for k in LABELS}
    pairs = windows(S.build())
    texts = [t for t, _ in pairs]
    probs = (scores_onnx(ROOT / a.onnx, texts, ROOT / a.tokenizer, a.max_len) if a.onnx
             else scores_torch(ROOT / a.torch_dir, texts, a.max_len))
    out = report(list(zip([c for _, c in pairs], probs)), a.name, th)
    (ROOT / f"data/eval/SUITE_MODEL_{a.name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
