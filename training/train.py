"""
PayChat Multi-Intent Detector - Training Script

Model: DistilBERT fine-tuned for multi-label classification.
Task: detect which of 5 intents a chat message fires (one message can fire multiple).

Intents (independent sigmoid heads):
  money | alarm | contact | calendar | maps

What this does:
  - Loads the multi-label dataset from generate_data.py
  - Fine-tunes DistilBERT with BCEWithLogitsLoss (5 heads, independent)
  - Per-intent precision/recall/F1 and ROC-AUC
  - Saves model + tokenizer + training_report.json
"""

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# ── CLI ──
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data-dir",   default=None)
_parser.add_argument("--output-dir", default=None)
_parser.add_argument("--epochs",     type=int, default=None)
_cli, _ = _parser.parse_known_args()

# ── Config ──
MODEL_NAME = "distilbert-base-uncased"
DATA_DIR   = Path(_cli.data_dir)   if _cli.data_dir   else Path(".")
OUT_DIR    = Path(_cli.output_dir) if _cli.output_dir else Path("../saved_model")
BATCH_SIZE = 16
EPOCHS     = _cli.epochs if _cli.epochs else 5
LR         = 2e-5
MAX_LEN    = 128
SEED       = 42
THRESHOLD  = 0.5   # per-intent sigmoid cutoff

INTENTS = ["money", "alarm", "contact", "calendar", "maps"]
NUM_LABELS = len(INTENTS)

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Intents: {INTENTS}")
print(f"Training with {EPOCHS} epochs, batch size {BATCH_SIZE}, lr {LR}, threshold {THRESHOLD}")


# ═════════════════════════════════════════════════════════════════════
#  Dataset
# ═════════════════════════════════════════════════════════════════════

class ChatDataset(Dataset):
    """Multi-label chat dataset. Each item returns a float label vector of size NUM_LABELS."""

    def __init__(self, items, tokenizer):
        self.items = items
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # labels dict -> float vector [money, alarm, contact, calendar, maps]
        label_vec = [float(item["labels"][intent]) for intent in INTENTS]
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label_vec, dtype=torch.float),
            "text":           item["text"],
            "category":       item.get("category", "unknown"),
        }


def load_split(split_name):
    path = DATA_DIR / f"{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}\nRun generate_data.py first!")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════
#  Train / Eval
# ═════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    all_texts  = []
    all_cats   = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_texts.extend(batch["text"])
            all_cats.extend(batch["category"])

    all_logits = np.concatenate(all_logits, axis=0)    # (N, 5)
    all_labels = np.concatenate(all_labels, axis=0)    # (N, 5)
    all_probs  = 1.0 / (1.0 + np.exp(-all_logits))     # sigmoid
    all_preds  = (all_probs >= THRESHOLD).astype(int)

    # Exact-match accuracy: all 5 intents match
    exact_match = np.all(all_preds == all_labels, axis=1).mean()
    # Hamming accuracy: fraction of (example, intent) pairs correct
    hamming_acc = (all_preds == all_labels).mean()
    avg_loss = total_loss / len(loader)

    return {
        "loss":        avg_loss,
        "exact_match": exact_match,
        "hamming_acc": hamming_acc,
        "probs":       all_probs,
        "preds":       all_preds,
        "labels":      all_labels,
        "texts":       all_texts,
        "cats":        all_cats,
    }


def per_intent_report(results):
    """Precision/Recall/F1/AUC for each intent."""
    print("\n  Per-Intent Metrics (threshold = {:.2f}):".format(THRESHOLD))
    print(f"    {'intent':<10} {'precision':>10} {'recall':>8} {'f1':>8} {'auc':>8}  {'support'}")

    summary = {}
    for i, intent in enumerate(INTENTS):
        y_true = results["labels"][:, i].astype(int)
        y_pred = results["preds"][:, i].astype(int)
        y_prob = results["probs"][:, i]

        if y_true.sum() == 0:
            p = r = f = auc = 0.0
        else:
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_prob) if y_true.sum() > 0 and y_true.sum() < len(y_true) else 0.0
            except ValueError:
                auc = 0.0

        support = int(y_true.sum())
        print(f"    {intent:<10} {p:>9.1%} {r:>7.1%} {f:>7.1%} {auc:>7.4f}    {support}")
        summary[intent] = {
            "precision": float(p),
            "recall":    float(r),
            "f1":        float(f),
            "auc":       float(auc),
            "support":   support,
        }
    return summary


def find_errors(results, n=15):
    """Surface hardest errors — examples where the predicted set doesn't match the true set."""
    errors = []
    for i, (text, cat) in enumerate(zip(results["texts"], results["cats"])):
        pred_set = set(INTENTS[j] for j, v in enumerate(results["preds"][i]) if v == 1)
        true_set = set(INTENTS[j] for j, v in enumerate(results["labels"][i]) if v == 1)
        if pred_set != true_set:
            errors.append({
                "text": text,
                "true": sorted(true_set) or ["none"],
                "predicted": sorted(pred_set) or ["none"],
                "probs": {INTENTS[j]: float(results["probs"][i][j]) for j in range(NUM_LABELS)},
                "category": cat,
            })
    # sort by ambiguity (probs closest to 0.5)
    errors.sort(key=lambda e: min(abs(p - 0.5) for p in e["probs"].values()))
    return errors[:n]


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  PayChat Multi-Intent Detector - Training")
    print("=" * 60 + "\n")

    print("Loading data...")
    train_items = load_split("train")
    val_items   = load_split("val")
    test_items  = load_split("test")
    print(f"  Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = ChatDataset(train_items, tokenizer)
    val_ds   = ChatDataset(val_items,   tokenizer)
    test_ds  = ChatDataset(test_items,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Multi-label model: HF applies BCEWithLogitsLoss automatically when problem_type is set
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label={i: intent for i, intent in enumerate(INTENTS)},
        label2id={intent: i for i, intent in enumerate(INTENTS)},
    )
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    print(f"\nTraining for {EPOCHS} epochs...\n", flush=True)
    best_val_hamming = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f"  Epoch {epoch}/{EPOCHS} starting...", flush=True)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_results = evaluate(model, val_loader)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_results['loss']:.4f} | "
              f"Val Exact: {val_results['exact_match']:.1%} | "
              f"Val Hamming: {val_results['hamming_acc']:.1%} | "
              f"Time: {elapsed:.1f}s", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_results["loss"],
            "val_exact_match": val_results["exact_match"],
            "val_hamming": val_results["hamming_acc"],
        })

        if val_results["hamming_acc"] > best_val_hamming:
            best_val_hamming = val_results["hamming_acc"]
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
            print(f"  -> New best saved (hamming {best_val_hamming:.1%})", flush=True)

    # ── Final test ──
    print("\n" + "=" * 60)
    print("  Final Test Set Evaluation")
    print("=" * 60)
    test_results = evaluate(model, test_loader)

    print(f"\n  Exact-match accuracy (all 5 intents right): {test_results['exact_match']:.2%}")
    print(f"  Hamming accuracy    (per-cell correctness): {test_results['hamming_acc']:.2%}")

    intent_summary = per_intent_report(test_results)

    print("\n  Hardest errors (most ambiguous):")
    for err in find_errors(test_results, n=12):
        probs_str = ", ".join(f"{k}={v:.2f}" for k, v in err["probs"].items())
        print(f"    [true={err['true']} pred={err['predicted']}] \"{err['text'][:60]}\"")
        print(f"       probs: {probs_str}")

    report = {
        "trained_at":     datetime.utcnow().isoformat(),
        "model":          MODEL_NAME,
        "epochs":         EPOCHS,
        "intents":        INTENTS,
        "threshold":      THRESHOLD,
        "test_exact_match": test_results["exact_match"],
        "test_hamming":     test_results["hamming_acc"],
        "per_intent":       intent_summary,
        "training_history": history,
    }
    with open(OUT_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Model saved to {OUT_DIR}/")
    print(f"  Report saved to {OUT_DIR}/training_report.json")
    print(f"\n{'=' * 60}")
    print(f"  TEST EXACT-MATCH: {test_results['exact_match']:.2%}")
    print(f"  TEST HAMMING:     {test_results['hamming_acc']:.2%}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
