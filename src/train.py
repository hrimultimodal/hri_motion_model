"""
train.py

Training loop for MotionLSTM.

Usage:
    python train.py

Outputs:
    checkpoints/best_model.pt     ← best val accuracy checkpoint
    checkpoints/last_model.pt     ← final epoch checkpoint
    logs/training_log.json        ← full loss/accuracy history
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from model import MotionLSTM

# ─── Paths ────────────────────────────────────────────────────────────────────
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(SRC_DIR, "..", "data",        "processed")
CKPT_DIR  = os.path.join(SRC_DIR, "..", "checkpoints")
LOG_DIR   = os.path.join(SRC_DIR, "..", "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
CONFIG = {
    "batch_size":   128,
    "epochs":       150,
    "lr":           0.0022097021477741094,
    "weight_decay": 0.0043583865743590305,
    "hidden_size":  256,
    "num_layers":   3,
    "dropout":      0.35,
    "patience":     20,
}

MOTION_LABELS = {
    0: "sitting",
    1: "standing",
    2: "walking",
    3: "running",
    4: "stepping_back",
    5: "slumped",
}

# ─── Dataset ──────────────────────────────────────────────────────────────────

class MotionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    print(f"  Train : X={X_train.shape}  y={y_train.shape}")
    print(f"  Val   : X={X_val.shape}    y={y_val.shape}")

    # Class distribution
    counts = Counter(y_train.tolist())
    print("\n  Class distribution (train):")
    for idx, name in MOTION_LABELS.items():
        print(f"    {idx} {name:<15}: {counts.get(idx, 0):>7,}")

    return X_train, y_train, X_val, y_val


def compute_class_weights(y_train: np.ndarray, device: torch.device):
    """
    Compute per-class weights inversely proportional to class frequency.
    Passed to CrossEntropyLoss so minority classes (running, stepping_back)
    contribute equally to the loss as majority classes (sitting).
    """
    classes = np.arange(len(MOTION_LABELS))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    print(f"\n  Class weights (higher = model penalised more for errors):")
    for idx, name in MOTION_LABELS.items():
        print(f"    {idx} {name:<15}: {weights[idx]:.3f}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


def evaluate(model, loader, criterion, device):
    """
    Run one full pass over a dataloader.
    Returns (avg_loss, accuracy, all_preds, all_labels).
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, accuracy, all_preds, all_labels


# ─── Main training loop ───────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training MotionLSTM")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val = load_data()

    train_ds = MotionDataset(X_train, y_train)
    val_ds   = MotionDataset(X_val,   y_val)

    # Oversample minority classes so each epoch sees a more balanced mix.
    # stepping_back and running had the lowest counts — this gives them
    # proportionally more appearances per epoch without duplicating data
    # in a fixed way (sampler draws randomly with replacement).
    from torch.utils.data import WeightedRandomSampler

    label_counts  = Counter(y_train.tolist())
    sample_weights = [
        1.0 / label_counts[int(label)] for label in y_train
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,        # replaces shuffle=True
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = compute_class_weights(y_train, device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MotionLSTM(
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # CosineAnnealingLR: smoothly decays LR from `lr` to near 0 over all epochs
    # Better than StepLR for this size of dataset
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Note'}")
    print(f"{'─'*60}")

    best_val_acc  = 0.0
    patience_ctr  = 0
    log_history   = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        # ── Train one epoch ───────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping — prevents exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss    += loss.item()
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += len(y_batch)

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        train_acc      = train_correct / train_total

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        # ── Checkpoint + early stopping ───────────────────────────────────────
        note = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":    val_acc,
                    "config":     CONFIG,
                },
                os.path.join(CKPT_DIR, "best_model.pt"),
            )
            note = "← best"
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {CONFIG['patience']} epochs)")
                break

        # ── Log ───────────────────────────────────────────────────────────────
        log_history.append({
            "epoch":      epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_acc":  round(train_acc,      4),
            "val_loss":   round(val_loss,        4),
            "val_acc":    round(val_acc,         4),
            "lr":         round(current_lr,      6),
        })

        print(f"{epoch:>6}  {avg_train_loss:>10.4f}  {train_acc:>8.3%}  "
              f"{val_loss:>8.4f}  {val_acc:>7.3%}  "
              f"{current_lr:>8.6f}  {note}")

    # ── Save last model ───────────────────────────────────────────────────────
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(CKPT_DIR, "last_model.pt"),
    )

    # ── Final evaluation on val set (using best checkpoint) ──────────────────
    print(f"\n{'='*60}")
    print(f"Best val accuracy : {best_val_acc:.3%}")
    print(f"Loading best checkpoint for final report...")

    checkpoint = torch.load(os.path.join(CKPT_DIR, "best_model.pt"),
                            weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, val_preds, val_labels = evaluate(
        model, val_loader, criterion, device
    )

    label_names = [MOTION_LABELS[i] for i in range(len(MOTION_LABELS))]

    print(f"\nPer-class report (val set):")
    print(classification_report(val_labels, val_preds,
                                 target_names=label_names, digits=3))

    print(f"Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(val_labels, val_preds)
    print(f"{'':>15}", end="")
    for name in label_names:
        print(f"{name[:8]:>10}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{label_names[i]:>15}", end="")
        for val in row:
            print(f"{val:>10}", end="")
        print()

    # ── Save logs ─────────────────────────────────────────────────────────────
    with open(os.path.join(LOG_DIR, "training_log.json"), "w") as f:
        json.dump({"config": CONFIG, "history": log_history}, f, indent=2)
    print(f"\nLogs saved to: {LOG_DIR}/training_log.json")
    print(f"Best model  : {CKPT_DIR}/best_model.pt")


if __name__ == "__main__":
    train()