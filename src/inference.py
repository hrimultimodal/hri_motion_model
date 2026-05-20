"""
inference.py

Real-time inference class for MotionLSTM.
Maintains a sliding window buffer internally.
Accepts one skeleton frame at a time, returns a prediction
every time the window is full.

Used by:
  - webcam_demo.py        (testing)
  - ROS2 motion node      (deployment)
  - fusion engine         (final integration)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Optional
from model import MotionLSTM

# ─── Constants (must match build_dataset.py exactly) ─────────────────────────
WINDOW_SIZE  = 30
FEATURE_DIM  = 84
NUM_CLASSES  = 6

MOTION_LABELS = {
    0: "sitting",
    1: "standing",
    2: "walking",
    3: "running",
    4: "stepping_back",
    5: "slumped",
}

# Emoji for webcam display
MOTION_EMOJI = {
    0: "🪑 sitting",
    1: "🧍 standing",
    2: "🚶 walking",
    3: "🏃 running",
    4: "⬅️  stepping back",
    5: "😔 slumped",
}

# Same 14 joint indices used during training
JOINT_SUBSET = [0, 1, 2, 3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17]
HIP_L_IDX      = JOINT_SUBSET.index(12)
HIP_R_IDX      = JOINT_SUBSET.index(16)
SHOULDER_L_IDX = JOINT_SUBSET.index(4)
SHOULDER_R_IDX = JOINT_SUBSET.index(8)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class MotionResult:
    label:       str           # e.g. "walking"
    label_idx:   int           # e.g. 2
    confidence:  float         # e.g. 0.91
    probs:       np.ndarray    # shape (6,) — full probability distribution
    stable:      bool          # True once window has been filled at least once

    def to_dict(self) -> dict:
        return {
            "label":      self.label,
            "label_idx":  self.label_idx,
            "confidence": round(self.confidence, 3),
            "probs":      {MOTION_LABELS[i]: round(float(p), 3)
                           for i, p in enumerate(self.probs)},
            "stable":     self.stable,
        }


# ─── Normalisation (identical to build_dataset.py) ───────────────────────────

def normalize_skeleton(joints: np.ndarray) -> np.ndarray:
    """
    joints: (14, 3)
    Returns hip-centered, shoulder-width normalised (14, 3)
    """
    hip_center = (joints[HIP_L_IDX] + joints[HIP_R_IDX]) / 2.0
    joints_norm = joints - hip_center
    shoulder_dist = np.linalg.norm(
        joints_norm[SHOULDER_L_IDX] - joints_norm[SHOULDER_R_IDX]
    )
    if shoulder_dist > 0.05:
        joints_norm = joints_norm / shoulder_dist
    return joints_norm.astype(np.float32)


# ─── Main inference class ─────────────────────────────────────────────────────

class MotionInference:
    """
    Stateful real-time inference engine.

    Usage:
        engine = MotionInference("checkpoints/best_model.pt")

        # In your frame loop:
        result = engine.update(joints_3d_25)   # pass full 25-joint array
        if result is not None:
            print(result.label, result.confidence)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        # ── Device ────────────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # ── Load model ────────────────────────────────────────────────────────
        checkpoint = torch.load(checkpoint_path,
                                map_location=self.device,
                                weights_only=True)
        cfg = checkpoint.get("config", {})

        self.model = MotionLSTM(
            hidden_size=cfg.get("hidden_size", 256),
            num_layers=cfg.get("num_layers",   3),
            dropout=cfg.get("dropout",         0.35),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # ── Sliding window buffer ─────────────────────────────────────────────
        # Stores normalised feature vectors, one per frame
        self._pos_buffer  = deque(maxlen=WINDOW_SIZE)  # positions
        self._prev_pos    = None                        # for velocity calc
        self._filled_once = False

        print(f"MotionInference ready | device={self.device} | "
              f"checkpoint epoch={checkpoint.get('epoch', '?')} | "
              f"val_acc={checkpoint.get('val_acc', 0):.3%}")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, joints_25: np.ndarray) -> Optional[MotionResult]:
        """
        Feed one frame of skeleton data.

        joints_25: numpy array of shape (25, 3)
                   Full 25-joint NTU/MediaPipe world landmarks in metres.
                   Pass zeros if no person detected — buffer will still
                   advance but prediction will be unreliable.

        Returns MotionResult if window is full, None while still filling.
        """
        # Subsample to 14 joints
        joints_14 = joints_25[JOINT_SUBSET]             # (14, 3)
        norm      = normalize_skeleton(joints_14)        # (14, 3)
        pos       = norm.flatten()                       # (42,)

        # Velocity relative to previous frame
        if self._prev_pos is None:
            vel = np.zeros(42, dtype=np.float32)
        else:
            vel = pos - self._prev_pos

        self._prev_pos = pos.copy()

        # Feature vector for this frame
        frame_feat = np.concatenate([pos, vel])         # (84,)
        self._pos_buffer.append(frame_feat)

        # Not enough frames yet
        if len(self._pos_buffer) < WINDOW_SIZE:
            return MotionResult(
                label="buffering",
                label_idx=-1,
                confidence=0.0,
                probs=np.zeros(NUM_CLASSES, dtype=np.float32),
                stable=False,
            )

        self._filled_once = True
        return self._predict()

    def reset(self):
        """Call when person leaves frame or scene changes."""
        self._pos_buffer.clear()
        self._prev_pos    = None
        self._filled_once = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _predict(self) -> MotionResult:
        window = np.stack(list(self._pos_buffer))      # (30, 84)
        tensor = torch.tensor(window, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).to(self.device)   # (1, 30, 84)

        with torch.no_grad():
            logits = self.model(tensor)                # (1, 6)
            probs  = F.softmax(logits, dim=-1)         # (1, 6)

        probs_np   = probs.squeeze(0).cpu().numpy()    # (6,)
        label_idx  = int(np.argmax(probs_np))
        confidence = float(probs_np[label_idx])

        return MotionResult(
            label=MOTION_LABELS[label_idx],
            label_idx=label_idx,
            confidence=confidence,
            probs=probs_np,
            stable=self._filled_once,
        )