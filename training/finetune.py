"""
Quick fine-tune from existing saved model with new training data.
Loads pre-trained weights (not base model) so converges in 2-3 epochs.
"""
import json
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Config
SAVED_MODEL = Path("../saved_model")
DATA_DIR = Path(".")
OUT_DIR = Path("../saved_model")
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-5  # lower LR since we're fine-tuning existing weights
MAX_LEN = 128
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class ChatDataset(Dataset):
    def __init__(self, items, tokenizer):
        self.items = items
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(item["text"], max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


def load_split(name):
    path = DATA_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

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

        if (i + 1) % 50 == 0:
            print(f"    Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}", flush=True)

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = (all_preds == all_labels).mean()
    auc = roc_auc_score(all_labels, all_probs)
    return {"accuracy": acc, "auc": auc, "preds": all_preds, "labels": all_labels, "probs": all_probs}


def main():
    print("\n" + "="*60)
    print("  PayChat — Fine-tuning from existing model")
    print("="*60 + "\n")

    # Load data
    train_items = load_split("train")
    val_items = load_split("val")
    test_items = load_split("test")
    print(f"Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

    # Load existing model + tokenizer
    print(f"\nLoading existing model from {SAVED_MODEL}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(SAVED_MODEL))
    model = DistilBertForSequenceClassification.from_pretrained(str(SAVED_MODEL))
    model = model.to(DEVICE)
    print("Loaded! (pre-trained weights, not base model)")

    # Datasets
    train_ds = ChatDataset(train_items, tokenizer)
    val_ds = ChatDataset(val_items, tokenizer)
    test_ds = ChatDataset(test_items, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    print(f"\nTraining for {EPOCHS} epochs (batch={BATCH_SIZE}, lr={LR})...\n")
    best_val_acc = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f"  Epoch {epoch}/{EPOCHS} starting...", flush=True)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_results = evaluate(model, val_loader)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%} | "
              f"Val Acc: {val_results['accuracy']:.1%} | Val AUC: {val_results['auc']:.4f} | "
              f"Time: {elapsed:.1f}s", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_results["accuracy"], 4),
            "val_auc": round(val_results["auc"], 4),
        })

        if val_results["accuracy"] >= best_val_acc:
            best_val_acc = val_results["accuracy"]
            # Save best model
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(OUT_DIR))
            tokenizer.save_pretrained(str(OUT_DIR))
            print(f"  -> Model saved to {OUT_DIR}", flush=True)

    # Final test evaluation
    print(f"\n{'='*60}")
    print("  Final Test Evaluation")
    print(f"{'='*60}\n")

    # Reload best model
    model = DistilBertForSequenceClassification.from_pretrained(str(OUT_DIR))
    model = model.to(DEVICE)
    test_results = evaluate(model, test_loader)

    cm = confusion_matrix(test_results["labels"], test_results["preds"])
    print(f"Test Accuracy: {test_results['accuracy']:.1%}")
    print(f"Test AUC: {test_results['auc']:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")
    print(classification_report(test_results["labels"], test_results["preds"], target_names=["NOT_MONEY", "MONEY"]))

    # Save report
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f1, _ = precision_recall_fscore_support(test_results["labels"], test_results["preds"], average="binary")
    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model": "distilbert-base-uncased (fine-tuned)",
        "epochs": EPOCHS,
        "test_accuracy": round(float(test_results["accuracy"]), 4),
        "test_auc": round(float(test_results["auc"]), 4),
        "test_precision": round(float(p), 4),
        "test_recall": round(float(r), 4),
        "test_f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "training_history": history,
    }
    with open(OUT_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDone! Model saved to {OUT_DIR}")
    print(f"Report saved to {OUT_DIR / 'training_report.json'}")


if __name__ == "__main__":
    main()
