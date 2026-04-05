"""
PayChat Money Detector — Training Script
Model: DistilBERT fine-tuned for binary classification
Task: Detect money-related messages in casual/transactional chat

What this does:
  - Loads your generated training data
  - Fine-tunes DistilBERT (67MB, fast, accurate)
  - Evaluates with precision/recall/F1 per category
  - Saves model + tokenizer for API use
  - Exports accuracy report so you know exactly how good it is
"""

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Imports (installed via requirements.txt) ──
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ── CLI args (used by continuous_learning/scheduler.py) ──
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data-dir",   default=None)
_parser.add_argument("--output-dir", default=None)
_parser.add_argument("--epochs",     type=int, default=None)
_cli, _ = _parser.parse_known_args()

# ── Config ──
MODEL_NAME = "distilbert-base-uncased"
DATA_DIR   = Path(_cli.data_dir)   if _cli.data_dir   else Path("../data")
OUT_DIR    = Path(_cli.output_dir) if _cli.output_dir else Path("./saved_model")
BATCH_SIZE = 32
EPOCHS     = _cli.epochs if _cli.epochs else 5   # increase to 8 for even better results
LR         = 2e-5
MAX_LEN    = 128      # chat messages are short; 128 is plenty
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Training with {EPOCHS} epochs, batch size {BATCH_SIZE}, lr {LR}")


# ── Dataset ──
class ChatDataset(Dataset):
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
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(item["label"], dtype=torch.long),
            "category":       item.get("category", "unknown"),
            "text":           item["text"],
        }


def load_split(split_name):
    path = DATA_DIR / f"{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}\nRun data/generate_data.py first!")
    with open(path) as f:
        return json.load(f)


# ── Training ──
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Evaluation ──
def evaluate(model, loader, split_name="val"):
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []
    all_cats   = []
    all_texts  = []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            all_cats.extend(batch["category"])
            all_texts.extend(batch["text"])

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    auc = roc_auc_score(all_labels, all_probs)
    avg_loss = total_loss / len(loader)

    return {
        "loss":     avg_loss,
        "accuracy": acc,
        "auc":      auc,
        "preds":    all_preds,
        "labels":   all_labels,
        "probs":    all_probs,
        "cats":     all_cats,
        "texts":    all_texts,
    }


def per_category_report(results):
    """Breakdown accuracy per money category (only for positive examples)."""
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, label, cat in zip(results["preds"], results["labels"], results["cats"]):
        if label == 1:  # only positives
            cat_stats[cat]["total"] += 1
            if pred == label:
                cat_stats[cat]["correct"] += 1

    print("\n  Per-Category Accuracy (positive examples only):")
    for cat, stats in sorted(cat_stats.items()):
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            bar = "#" * int(acc * 20)
            print(f"    {cat:<20} {acc:.1%}  {bar}  ({stats['correct']}/{stats['total']})", flush=True)


def find_errors(results, n=10):
    """Surface the worst prediction errors for analysis."""
    errors = []
    for text, pred, label, prob, cat in zip(
        results["texts"], results["preds"], results["labels"], results["probs"], results["cats"]
    ):
        if pred != label:
            errors.append({
                "text":       text,
                "true":       "MONEY" if label == 1 else "NOT_MONEY",
                "predicted":  "MONEY" if pred  == 1 else "NOT_MONEY",
                "confidence": prob,
                "category":   cat,
            })
    errors.sort(key=lambda x: abs(x["confidence"] - 0.5))  # closest to decision boundary
    return errors[:n]


# ── Main ──
def main():
    print("\n" + "="*60)
    print("  PayChat Money Detector — Training")
    print("="*60 + "\n")

    # Load data
    print("Loading data...")
    train_items = load_split("train")
    val_items   = load_split("val")
    test_items  = load_split("test")
    print(f"  Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

    # Tokenizer
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Datasets + loaders
    train_ds = ChatDataset(train_items, tokenizer)
    val_ds   = ChatDataset(val_items,   tokenizer)
    test_ds  = ChatDataset(test_items,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(DEVICE)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...\n", flush=True)
    best_val_acc = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f"  Epoch {epoch}/{EPOCHS} starting...", flush=True)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_results = evaluate(model, val_loader, "val")

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%} | "
              f"Val Acc: {val_results['accuracy']:.1%} | Val AUC: {val_results['auc']:.4f} | "
              f"Time: {elapsed:.1f}s", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_results["accuracy"],
            "val_auc": val_results["auc"],
        })

        # Save best model
        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
            print(f"  -> New best saved ({best_val_acc:.1%})", flush=True)

    # Final test evaluation
    print("\n" + "="*60)
    print("  Final Test Set Evaluation")
    print("="*60)
    test_results = evaluate(model, test_loader, "test")

    print(f"\n  Accuracy:  {test_results['accuracy']:.2%}")
    print(f"  AUC-ROC:   {test_results['auc']:.4f}")

    labels     = test_results["labels"]
    preds      = test_results["preds"]
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary")
    print(f"  Precision: {p:.2%}")
    print(f"  Recall:    {r:.2%}")
    print(f"  F1:        {f:.2%}")

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(f"    True Neg: {cm[0][0]:4d}  False Pos: {cm[0][1]:4d}")
    print(f"    False Neg: {cm[1][0]:3d}  True Pos:  {cm[1][1]:4d}")

    per_category_report(test_results)

    print("\n  Worst Errors (closest to decision boundary):")
    for err in find_errors(test_results):
        conf = err["confidence"]
        print(f"    [{err['true']} → {err['predicted']} | conf={conf:.2f}] \"{err['text'][:60]}\"")

    # Save report
    report = {
        "trained_at": datetime.utcnow().isoformat(),
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "test_accuracy": test_results["accuracy"],
        "test_auc": test_results["auc"],
        "test_precision": p,
        "test_recall": r,
        "test_f1": f,
        "confusion_matrix": cm.tolist(),
        "training_history": history,
    }
    with open(OUT_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Model saved to {OUT_DIR}/")
    print(f"✓ Training report saved to {OUT_DIR}/training_report.json")
    print(f"\n{'='*60}")
    print(f"  FINAL ACCURACY: {test_results['accuracy']:.2%}")
    print(f"  FINAL F1:       {f:.2%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
