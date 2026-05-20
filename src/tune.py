"""
tune.py

Bayesian hyperparameter search using Optuna.
Runs N trials, each training for a limited number of epochs,
and finds the config that best closes the train/val gap.

Usage:
    python tune.py

Output:
    logs/optuna_study.db        ← full study database
    logs/best_hparams.json      ← best config found
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna.samplers import TPESampler
from model import MotionLSTM

# ─── Paths ────────────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "..", "data",  "processed")
LOG_DIR  = os.path.join(SRC_DIR, "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Tuning config ────────────────────────────────────────────────────────────
N_TRIALS       = 40    # number of hyperparameter combinations to try
EPOCHS_PER_TRIAL = 40  # epochs per trial — enough to see convergence trend
                       # without spending 150 epochs × 40 trials
PATIENCE       = 8     # early stopping within each trial

MOTION_LABELS = {
    0: "sitting",    1: "standing", 2: "walking",
    3: "running",    4: "stepping_back", 5: "slumped",
}


# ─── Dataset (loaded once, reused across all trials) ─────────────────────────
class MotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):          return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    return X_train, y_train, X_val, y_val


# ─── Objective function ───────────────────────────────────────────────────────

def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    """
    Optuna calls this function for each trial.
    Returns val_accuracy (higher = better).

    Search space rationale:
      hidden_size  : bigger = more capacity, but more overfitting risk
      num_layers   : 3 layers might capture longer temporal patterns
      dropout      : currently 0.3 — search wider to fix the 5.6% gap
      weight_decay : L2 regularisation — currently 1e-4, try stronger
      lr           : current 1e-3 might be slightly high
      batch_size   : smaller batches = noisier gradients = implicit regularisation
    """

    # ── Suggest hyperparameters ───────────────────────────────────────────────
    hidden_size  = trial.suggest_categorical("hidden_size",  [64, 128, 256])
    num_layers   = trial.suggest_int(        "num_layers",   1, 3)
    dropout      = trial.suggest_float(      "dropout",      0.2, 0.6, step=0.05)
    lr           = trial.suggest_float(      "lr",           5e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float(      "weight_decay", 1e-5, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size",   [128, 256, 512])

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_ds = MotionDataset(X_train, y_train)
    val_ds   = MotionDataset(X_val,   y_val)

    label_counts   = Counter(y_train.tolist())
    sample_weights = [1.0 / label_counts[int(l)] for l in y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MotionLSTM(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    # ── Loss + optimiser ──────────────────────────────────────────────────────
    classes = np.arange(len(MOTION_LABELS))
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_PER_TRIAL
    )

    # ── Training loop (short) ─────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_ctr = 0

    for epoch in range(1, EPOCHS_PER_TRIAL + 1):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds    = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total   += len(y_batch)

        val_acc = correct / total

        # Report to Optuna — allows pruning of bad trials early
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping within trial
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return best_val_acc


# ─── Run study ────────────────────────────────────────────────────────────────

def run_tuning():
    print(f"\n{'='*60}")
    print(f"Optuna Hyperparameter Search")
    print(f"Trials : {N_TRIALS}  |  Epochs/trial : {EPOCHS_PER_TRIAL}")
    print(f"Device : {DEVICE}")
    print(f"{'='*60}\n")

    X_train, y_train, X_val, y_val = load_data()

    # TPE sampler = Tree-structured Parzen Estimator
    # Builds a probabilistic model of which configs perform well
    # and samples more from promising regions
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),

        # MedianPruner stops trials that are clearly underperforming
        # by epoch 15 — saves time on bad configs
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8,    # don't prune first 8 trials
            n_warmup_steps=15,     # don't prune before epoch 15
        ),

        # Save to disk — you can resume if interrupted
        storage=f"sqlite:///{os.path.join(LOG_DIR, 'optuna_study.db')}",
        study_name="motion_lstm_tuning",
        load_if_exists=True,
    )

    # Suppress per-trial Optuna logs — we print our own summary
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    completed = 0
    pruned    = 0

    for trial_num in range(N_TRIALS):
        trial = study.ask()
        try:
            val_acc = objective(trial, X_train, y_train, X_val, y_val)
            study.tell(trial, val_acc)
            completed += 1

            # Print every trial result
            best = study.best_value
            marker = " ← best" if abs(val_acc - best) < 1e-6 else ""
            print(f"Trial {trial_num+1:>3}/{N_TRIALS}  "
                  f"val_acc={val_acc:.4f}{marker}  "
                  f"| hidden={trial.params.get('hidden_size')} "
                  f"layers={trial.params.get('num_layers')} "
                  f"dropout={trial.params.get('dropout'):.2f} "
                  f"lr={trial.params.get('lr'):.5f} "
                  f"wd={trial.params.get('weight_decay'):.5f} "
                  f"bs={trial.params.get('batch_size')}")

        except optuna.exceptions.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            pruned += 1
            print(f"Trial {trial_num+1:>3}/{N_TRIALS}  PRUNED")

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Search complete: {completed} completed, {pruned} pruned")
    print(f"\nBest val accuracy : {study.best_value:.4%}")
    print(f"Best parameters   :")
    for k, v in study.best_params.items():
        print(f"  {k:<15}: {v}")

    # ── Save best config ──────────────────────────────────────────────────────
    best_config = {
        "val_acc_in_tuning": study.best_value,
        "epochs":       150,
        "patience":     20,
        **study.best_params,
    }
    out_path = os.path.join(LOG_DIR, "best_hparams.json")
    with open(out_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSaved to: {out_path}")

    # ── Top 5 trials ──────────────────────────────────────────────────────────
    print(f"\nTop 5 trials:")
    print(f"{'Rank':>4}  {'Val Acc':>8}  {'hidden':>6}  "
          f"{'layers':>6}  {'dropout':>7}  {'lr':>8}  {'wd':>8}  {'bs':>4}")
    print("─" * 65)
    trials_sorted = sorted(
        [t for t in study.trials
         if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value, reverse=True
    )[:5]
    for rank, t in enumerate(trials_sorted, 1):
        p = t.params
        print(f"{rank:>4}  {t.value:>8.4%}  {p['hidden_size']:>6}  "
              f"{p['num_layers']:>6}  {p['dropout']:>7.2f}  "
              f"{p['lr']:>8.5f}  {p['weight_decay']:>8.5f}  "
              f"{p['batch_size']:>4}")


if __name__ == "__main__":
    run_tuning()