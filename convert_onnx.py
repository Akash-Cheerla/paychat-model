"""
Convert the fine-tuned RoBERTa model to ONNX format with optional INT8 quantization.

Usage:
    python convert_onnx.py                    # ONNX only
    python convert_onnx.py --quantize         # ONNX + INT8 quantization
    python convert_onnx.py --validate         # Convert + run accuracy comparison

Output:
    saved_model_onnx/model.onnx              (or model_quantized.onnx)
    saved_model_onnx/tokenizer files          (copied from saved_model)
    saved_model_onnx/thresholds.json          (copied)
    saved_model_onnx/training_report.json     (copied)
    saved_model_onnx/config.json              (copied)
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "saved_model"
DST_DIR = ROOT / "saved_model_onnx"


def export_onnx(src: Path, dst: Path) -> Path:
    """Export PyTorch model to ONNX."""
    print(f"[onnx] Loading model from {src} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(src), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(src)).eval()

    # Create dummy input
    dummy = tokenizer("venmo me 30 for pizza", return_tensors="pt", max_length=128, truncation=True)

    dst.mkdir(parents=True, exist_ok=True)
    onnx_path = dst / "model.onnx"

    print(f"[onnx] Exporting to {onnx_path} ...")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,  # use legacy exporter (more reliable)
    )

    # Copy tokenizer + config files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "vocab.json", "merges.txt", "special_tokens_map.json",
                   "thresholds.json", "training_report.json"]:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, dst / fname)

    orig_size = sum(f.stat().st_size for f in src.iterdir()) / 1e6
    onnx_size = onnx_path.stat().st_size / 1e6
    print(f"[onnx] Original model: {orig_size:.1f} MB")
    print(f"[onnx] ONNX model:     {onnx_size:.1f} MB")
    print(f"[onnx] Reduction:      {orig_size/onnx_size:.1f}x smaller")

    return onnx_path


def quantize_model(onnx_path: Path) -> Path:
    """Apply INT8 dynamic quantization to ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quant_path = onnx_path.parent / "model_quantized.onnx"
    print(f"[onnx] Quantizing to INT8 -> {quant_path} ...")

    quantize_dynamic(
        str(onnx_path),
        str(quant_path),
        weight_type=QuantType.QInt8,
    )

    orig = onnx_path.stat().st_size / 1e6
    quant = quant_path.stat().st_size / 1e6
    print(f"[onnx] ONNX model:      {orig:.1f} MB")
    print(f"[onnx] Quantized model: {quant:.1f} MB")
    print(f"[onnx] Reduction:       {orig/quant:.1f}x smaller")

    return quant_path


def validate(src: Path, dst: Path, use_quantized: bool = False):
    """Compare PyTorch vs ONNX predictions on test cases."""
    import onnxruntime as ort

    print("\n[validate] Loading PyTorch model ...")
    tokenizer = AutoTokenizer.from_pretrained(str(src), use_fast=True)
    pt_model = AutoModelForSequenceClassification.from_pretrained(str(src)).eval()

    with open(src / "thresholds.json") as f:
        thresholds = json.load(f)

    labels = list(pt_model.config.id2label.values()) if pt_model.config.id2label else list(thresholds.keys())

    model_file = "model_quantized.onnx" if use_quantized else "model.onnx"
    onnx_path = dst / model_file
    print(f"[validate] Loading ONNX model from {onnx_path} ...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    test_cases = [
        ("venmo me 30 for pizza", ["money"]),
        ("uber to the airport", ["ride"]),
        ("order chipotle", ["food_order"]),
        ("remind me at 6pm", ["alarm"]),
        ("book a table at nobu", ["reservation"]),
        ("how do i get to the airport", ["maps"]),
        ("play some drake", ["music"]),
        ("flights to bali in december", ["travel"]),
        ("buy new headphones", ["shopping"]),
        ("lol", []),
        ("what's up", []),
        ("haha nice", []),
        ("hey", []),
        ("sending you a dollar", ["money"]),
        ("call mom", ["contact"]),
        ("is it gonna rain today", ["weather"]),
        ("schedule a dentist appointment", ["health"]),
        ("pay the electric bill", ["bills"]),
        ("concert tickets for travis scott", ["tickets"]),
        ("watch succession tonight", ["video"]),
        ("finish the report", ["task"]),
        ("save this for later", ["note"]),
        ("meeting at 3pm tomorrow", ["calendar"]),
    ]

    pt_times = []
    onnx_times = []
    mismatches = []

    print(f"\n{'Text':<45} {'PyTorch':<20} {'ONNX':<20} {'Match'}")
    print("-" * 110)

    for text, expected in test_cases:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        # PyTorch inference
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_logits = pt_model(**enc).logits[0]
        pt_time = (time.perf_counter() - t0) * 1000
        pt_times.append(pt_time)
        pt_probs = torch.sigmoid(pt_logits).numpy()
        pt_fired = [labels[i] for i, p in enumerate(pt_probs) if p >= thresholds.get(labels[i], 0.5)]

        # ONNX inference
        t0 = time.perf_counter()
        onnx_out = session.run(None, {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
        })
        onnx_time = (time.perf_counter() - t0) * 1000
        onnx_times.append(onnx_time)
        onnx_probs = 1 / (1 + np.exp(-onnx_out[0][0]))  # sigmoid
        onnx_fired = [labels[i] for i, p in enumerate(onnx_probs) if p >= thresholds.get(labels[i], 0.5)]

        match = set(pt_fired) == set(onnx_fired)
        status = "OK" if match else "MISMATCH"
        if not match:
            mismatches.append((text, pt_fired, onnx_fired))

        print(f"{text:<45} {str(pt_fired):<20} {str(onnx_fired):<20} {status}")

    print(f"\n{'='*60}")
    print(f"PyTorch  avg: {sum(pt_times)/len(pt_times):.1f}ms  (min {min(pt_times):.1f}ms, max {max(pt_times):.1f}ms)")
    print(f"ONNX     avg: {sum(onnx_times)/len(onnx_times):.1f}ms  (min {min(onnx_times):.1f}ms, max {max(onnx_times):.1f}ms)")
    print(f"Speedup:      {sum(pt_times)/sum(onnx_times):.1f}x faster")
    print(f"\nAccuracy: {len(test_cases)-len(mismatches)}/{len(test_cases)} match ({(len(test_cases)-len(mismatches))/len(test_cases)*100:.1f}%)")

    if mismatches:
        print(f"\nMISMATCHES:")
        for text, pt, onnx in mismatches:
            print(f"  {text}: PyTorch={pt}, ONNX={onnx}")
        return False
    else:
        print(f"\nAll predictions match. ONNX conversion is safe.")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FYOE model to ONNX")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization")
    parser.add_argument("--validate", action="store_true", help="Run accuracy comparison")
    args = parser.parse_args()

    onnx_path = export_onnx(SRC_DIR, DST_DIR)

    if args.quantize:
        quant_path = quantize_model(onnx_path)

    if args.validate:
        validate(SRC_DIR, DST_DIR, use_quantized=args.quantize)

    print(f"\n[done] ONNX model saved to {DST_DIR}")
    if args.quantize:
        print(f"[done] Use MODEL_DIR={DST_DIR} and set ONNX=1 to use it")
