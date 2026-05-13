"""One-shot baseline: run all seed_tests.jsonl through the v3 model, write a markdown report.

Usage:
    python eval/run_seed_baseline.py

Outputs:
    eval/baseline_report.md          — human-readable
    eval/baseline_report.json        — machine-readable
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from smart_postprocess import smart_suppress

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "saved_model"
SEED_FILE = ROOT / "eval" / "seed_tests.jsonl"
REPORT_MD = ROOT / "eval" / "baseline_report.md"
REPORT_JSON = ROOT / "eval" / "baseline_report.json"


def main() -> None:
    print(f"[base] loading model from {MODEL_DIR} ...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device).eval()
    with open(MODEL_DIR / "thresholds.json") as f:
        thresholds: dict[str, float] = json.load(f)
    labels: list[str] = list(mdl.config.id2label.values())

    try:
        with open(MODEL_DIR / "training_report.json") as f:
            training = json.load(f)
        test_exact = training.get("test_exact_match", 0) * 100
    except Exception:
        test_exact = 0.0

    print(f"[base] device={device} · {len(labels)} labels · test exact-match={test_exact:.2f}%", flush=True)

    cases = []
    with open(SEED_FILE, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                cases.append(json.loads(s))
    print(f"[base] loaded {len(cases)} seed cases", flush=True)

    # Inference (one at a time for clarity; tiny dataset)
    # v5: supports multi-turn context tests. If a case has a "context" field,
    # the tokenizer encodes (context, text) as a pair — matching training format.
    results = []
    t0 = time.time()
    for case in cases:
        context = case.get("context")
        if context:
            # Multi-turn: tokenizer pair encoding
            enc = tok(context, case["text"], return_tensors="pt", truncation=True, max_length=128).to(device)
        else:
            enc = tok(case["text"], return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = mdl(**enc).logits[0]
        probs = torch.sigmoid(logits).cpu().tolist()
        fired_raw = [labels[i] for i in range(len(labels)) if probs[i] >= thresholds.get(labels[i], 0.5)]
        # v5.1: smart post-processing — suppress false co-fires
        fired = smart_suppress(case["text"], fired_raw)
        expected = list(case.get("intents", []))
        results.append({
            "text": case["text"],
            "context": context,
            "tag": case.get("tag", "untagged"),
            "expected": expected,
            "fired": fired,
            "match": set(fired) == set(expected),
            "extra": sorted(set(fired) - set(expected)),
            "missed": sorted(set(expected) - set(fired)),
            "top5": sorted(
                [(labels[i], probs[i], thresholds.get(labels[i], 0.5)) for i in range(len(labels))],
                key=lambda x: -x[1],
            )[:5],
        })
    elapsed = time.time() - t0
    ms_per = elapsed * 1000 / max(1, len(cases))

    total = len(results)
    passed = sum(1 for r in results if r["match"])
    failed = total - passed

    # By tag
    by_tag: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0, "failures": []})
    for r in results:
        by_tag[r["tag"]]["total"] += 1
        if r["match"]:
            by_tag[r["tag"]]["passed"] += 1
        else:
            by_tag[r["tag"]]["failures"].append(r)

    # Per-intent
    per_intent = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    for r in results:
        exp_s, fir_s = set(r["expected"]), set(r["fired"])
        for label in labels:
            ip, ie = label in fir_s, label in exp_s
            if ip and ie:
                per_intent[label]["tp"] += 1
            elif ip:
                per_intent[label]["fp"] += 1
            elif ie:
                per_intent[label]["fn"] += 1

    print(f"[base] done in {elapsed:.1f}s ({ms_per:.0f}ms/case)")
    print(f"[base] {passed}/{total} passed ({passed/total*100:.1f}%) · {failed} failures")

    # ---- write JSON ----
    REPORT_JSON.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_dir": str(MODEL_DIR),
        "device": device,
        "test_exact_match": test_exact,
        "total": total,
        "passed": passed,
        "failed": failed,
        "ms_per_case": ms_per,
        "by_tag": {k: {"total": v["total"], "passed": v["passed"]} for k, v in by_tag.items()},
        "per_intent": per_intent,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- write markdown ----
    out = []
    out.append("# FYOE v3 — seed baseline report\n")
    out.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M')}  ")
    out.append(f"**Model:** roberta-base v3 ({sum(p.numel() for p in mdl.parameters())/1e6:.0f}M params, device: `{device}`)  ")
    out.append(f"**IID test-set accuracy (training distribution):** {test_exact:.2f}%  ")
    out.append(f"**Seed-suite size:** {total} curated real-world cases  ")
    out.append(f"**Latency:** {ms_per:.0f} ms / case (single-batch CPU)\n")

    out.append("## TL;DR\n")
    out.append(f"- **{passed}/{total} passed ({passed/total*100:.1f}%)** on the seed suite")
    out.append(f"- **{failed} failures** — these become v4 training data")
    pct_drop = test_exact - (passed / total * 100)
    out.append(f"- Reality gap: **{pct_drop:.1f} percentage points** between IID test set and adversarial seed suite")
    out.append("- This gap is the work item. Closing it = production-ready model.\n")

    out.append("## Performance by category\n")
    out.append("| Tag | Cases | Passed | Pass rate | Status |")
    out.append("|---|---:|---:|---:|---|")
    tag_order = [
        "smoke", "multi_intent", "numerics", "slang", "typo", "long",
        "negation", "past_tense", "ambiguous", "chitchat", "false_positive_bait",
        "code_mixed",
        "edge_pronoun", "edge_negation_marker", "edge_multi_recipient", "edge_missing_slot",
        "alarm_with_note", "alarm_missing_note", "alarm_missing_all",
        "maps_simple", "maps_query_not_action",
        "money_split", "query_not_action", "health_implicit", "weather_minimal",
        # v5 multi-turn context
        "v5_context_buildup", "v5_confirmation", "v5_confirm_negative",
        "v5_context_negative", "v5_correction", "v5_no_carryforward",
        "v5_disambiguation", "v5_single_with_context",
    ]
    seen_tags = set()
    for t in tag_order + [x for x in by_tag if x not in tag_order]:
        if t not in by_tag or t in seen_tags:
            continue
        seen_tags.add(t)
        info = by_tag[t]
        rate = info["passed"] / info["total"] * 100
        if rate >= 100:
            status = "PERFECT"
        elif rate >= 80:
            status = "minor gaps"
        elif rate >= 50:
            status = "needs work"
        else:
            status = "CRITICAL"
        out.append(f"| `{t}` | {info['total']} | {info['passed']} | {rate:.0f}% | {status} |")
    out.append("")

    out.append(f"## All {failed} failures (grouped by category)\n")
    seen_tags = set()
    for t in tag_order + [x for x in by_tag if x not in tag_order]:
        if t not in by_tag or t in seen_tags:
            continue
        seen_tags.add(t)
        fails = by_tag[t]["failures"]
        if not fails:
            continue
        out.append(f"### `{t}` — {len(fails)} failure{'s' if len(fails) != 1 else ''}\n")
        for f in fails:
            exp = ", ".join(f["expected"]) if f["expected"] else "(none / chitchat)"
            fired = ", ".join(f["fired"]) if f["fired"] else "(none)"
            top = " · ".join(f"`{n}`={p:.2f}" for n, p, _ in f["top5"])
            extras = f", **extra:** {', '.join(f['extra'])}" if f["extra"] else ""
            missed = f", **missed:** {', '.join(f['missed'])}" if f["missed"] else ""
            ctx_str = f"  context: `{f['context']}`  \n" if f.get("context") else ""
            out.append(f"- `{f['text']}`  ")
            if ctx_str:
                out.append(ctx_str)
            out.append(f"  expected **{exp}** · fired **{fired}**{extras}{missed}  ")
            out.append(f"  top probs: {top}")
        out.append("")

    out.append("## Per-intent precision / recall (on seed suite)\n")
    out.append("| Intent | TP | FP | FN | Precision | Recall | F1 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        m = per_intent[label]
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = 2 * p * r / (p + r) if p and r else None
        ps = f"{p*100:.0f}%" if p is not None else "—"
        rs = f"{r*100:.0f}%" if r is not None else "—"
        fs = f"{f1*100:.0f}%" if f1 is not None else "—"
        out.append(f"| `{label}` | {tp} | {fp} | {fn} | {ps} | {rs} | {fs} |")
    out.append("")

    # Recommendations
    weak = [(t, by_tag[t]["passed"] / by_tag[t]["total"]) for t in by_tag]
    weak.sort(key=lambda x: x[1])
    out.append("## v4 work list (weakest categories first)\n")
    for t, rate in weak:
        if rate >= 1.0:
            continue
        info = by_tag[t]
        gap = info["total"] - info["passed"]
        out.append(f"- **`{t}`** — {info['passed']}/{info['total']} passed. Add ~{max(50, gap*30)} targeted training examples + reweight loss for these patterns.")
    out.append("")

    REPORT_MD.write_text("\n".join(out), encoding="utf-8")
    print(f"[base] wrote {REPORT_MD}")
    print(f"[base] wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
