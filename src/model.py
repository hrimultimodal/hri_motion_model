"""
model.py

Lightweight 2-layer LSTM with attention pooling for motion classification.

Input:  (batch, seq_len=30, features=84)
Output: (batch, num_classes=6)

Design constraints (Jetson Orin Nano co-deployment):
  - Parameters  : ~270k
  - Model size  : ~1.1 MB
  - Inference   : <3ms per window
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionLSTM(nn.Module):
    def __init__(
        self,
        input_size:  int = 84,
        hidden_size: int = 128,
        num_layers:  int = 2,
        num_classes: int = 6,
        dropout:     float = 0.4,
    ):
        super().__init__()

        # ── 1. Input normalisation ──────────────────────────────────────────
        # LayerNorm on the feature dimension stabilises training when
        # velocity features have very different scales to position features.
        self.input_norm = nn.LayerNorm(input_size)

        # ── 2. LSTM ─────────────────────────────────────────────────────────
        # batch_first=True → input shape is (B, T, F) not (T, B, F)
        # dropout between LSTM layers (only applied when num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── 3. Temporal attention ───────────────────────────────────────────
        # Instead of using only the last hidden state, we let the model
        # learn WHICH timesteps in the window matter most.
        # e.g. for "running" the peak velocity frame matters more than
        # the first frame where the person is just starting to move.
        self.attention = nn.Linear(hidden_size, 1)

        # ── 4. Classifier head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 30, 84)
        returns: (B, 6)  raw logits — NOT softmax
        """
        # Normalise input features
        x = self.input_norm(x)                          # (B, 30, 84)

        # Run through LSTM
        lstm_out, _ = self.lstm(x)                      # (B, 30, 128)

        # Compute attention weights over the time dimension
        # attention: (B, 30, 1) → softmax → probability over 30 timesteps
        attn_weights = F.softmax(
            self.attention(lstm_out), dim=1
        )                                               # (B, 30, 1)

        # Weighted sum across time → single context vector per sample
        context = (attn_weights * lstm_out).sum(dim=1)  # (B, 128)

        # Classify
        return self.classifier(context)                 # (B, 6)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method — returns softmax probabilities."""
        return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MotionLSTM()
    print(f"Parameters : {model.count_parameters():,}")

    # Simulate one batch
    dummy = torch.zeros(32, 30, 84)
    out   = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print("Model architecture:")
    print(model)