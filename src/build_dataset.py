"""
build_dataset.py

Reads NTU RGB+D .skeleton files, extracts sliding-window features,
and saves processed train/val arrays to data/processed/.

Usage:
    python build_dataset.py

Output:
    data/processed/X_train.npy   shape: (N_train, 30, 75)
    data/processed/y_train.npy   shape: (N_train,)
    data/processed/X_val.npy     shape: (N_val,   30, 75)
    data/processed/y_val.npy     shape: (N_val,)
    data/processed/dataset_info.json
"""

import os
import sys
import json
import time
import numpy as np
from glob import glob
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
NTU_DIR  = os.environ.get(
    "NTU_DIR",
    os.path.expanduser(
        "~/Downloads/NTU_RGB Action Recognition Dataset"
        "/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons"
    )
)
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Windowing ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 30    # frames  (~1 second at 30 FPS)
STEP_SIZE   = 10    # stride  (67% overlap — gives 3× more samples)
MIN_FRAMES  = WINDOW_SIZE

# ─── Motion label definitions ─────────────────────────────────────────────────
MOTION_LABELS = {
    0: "sitting",
    1: "standing",
    2: "walking",
    3: "running",
    4: "stepping_back",
    5: "slumped",
}
NUM_CLASSES = len(MOTION_LABELS)

# ─── NTU action ID (1-indexed) → motion label ────────────────────────────────
#
# Mapping rationale:
#   sitting      A002(eat meal), A008(sit down), A011(reading),
#                A029(phone seated), A030(typing)
#   standing     A001(drink water), A009(stand up), A010(clapping),
#                A023(hand wave)
#   walking      A059(walk toward each other), A060(walk apart)
#   running      A026(hopping/jumping)
#   stepping_back A027(jump up — sudden recoil), A042(staggering)
#   slumped      A043(falling), A044(headache/pain), A045(stomachache)
#
NTU_TO_MOTION = {
    2:  0,   # eat meal            → sitting
    8:  0,   # sitting down        → sitting
    11: 0,   # reading             → sitting
    29: 0,   # playing with phone  → sitting
    30: 0,   # typing on keyboard  → sitting

    1:  1,   # drink water         → standing
    9:  1,   # standing up         → standing
    10: 1,   # clapping            → standing
    23: 1,   # hand waving         → standing

    59: 2,   # walk toward each other → walking
    60: 2,   # walk apart             → walking

    26: 3,   # hopping (one-foot jump) → running

    27: 4,   # jump up            → stepping_back
    42: 4,   # staggering         → stepping_back

    43: 5,   # falling            → slumped
    44: 5,   # headache/pain      → slumped
    45: 5,   # stomachache/pain   → slumped
}

# ─── Joints we keep (13 out of 25) ───────────────────────────────────────────
# Chosen to cover full body with minimum redundancy.
# Dropping finger tips, thumbs, feet tips — too noisy, not useful for motion.
JOINT_SUBSET = [
    0,   # base_of_spine   ← CoM proxy
    1,   # mid_spine
    2,   # neck
    3,   # head
    4,   # left_shoulder
    8,   # right_shoulder
    5,   # left_elbow
    9,   # right_elbow
    6,   # left_wrist
    10,  # right_wrist
    12,  # left_hip
    16,  # right_hip
    13,  # left_knee
    17,  # right_knee
]
NUM_JOINTS = len(JOINT_SUBSET)  # 14

# Hip indices within JOINT_SUBSET (for normalization)
HIP_L_IDX = JOINT_SUBSET.index(12)   # left_hip  in subset
HIP_R_IDX = JOINT_SUBSET.index(16)   # right_hip in subset
SHOULDER_L_IDX = JOINT_SUBSET.index(4)
SHOULDER_R_IDX = JOINT_SUBSET.index(8)

# Feature dim per frame:
# 14 joints × 3 (positions) + 14 joints × 3 (velocities) + 3 (CoM velocity)
# = 42 + 42 + 3 = 87  ... wait, let me simplify to keep it clean:
# positions: 14×3 = 42
# velocities: 14×3 = 42 (but first frame = 0)
# → total per frame: 84  (no CoM velocity separately — already in joint[0])
FEATURE_DIM = NUM_JOINTS * 3 * 2   # 84


# ─── Core functions ───────────────────────────────────────────────────────────

def parse_skeleton(path: str) -> np.ndarray | None:
    """
    Parse a .skeleton file.
    Returns array of shape (T, 25, 3) or None if file is empty/malformed.
    Only reads the FIRST body in each frame (single-person assumption).
    """
    try:
        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
    except Exception:
        return None

    try:
        idx = 0
        total_frames = int(lines[idx]); idx += 1
        frames = []

        for _ in range(total_frames):
            num_bodies = int(lines[idx]); idx += 1

            if num_bodies == 0:
                frames.append(np.zeros((25, 3), dtype=np.float32))
                continue

            # First body: skip the 1 metadata line
            idx += 1
            num_joints = int(lines[idx]); idx += 1

            joints = np.zeros((25, 3), dtype=np.float32)
            for j in range(min(num_joints, 25)):
                vals = lines[idx].split()
                joints[j, 0] = float(vals[0])
                joints[j, 1] = float(vals[1])
                joints[j, 2] = float(vals[2])
                idx += 1

            frames.append(joints)

            # Skip additional bodies
            for _ in range(num_bodies - 1):
                idx += 1               # body metadata
                nj = int(lines[idx]); idx += 1
                idx += nj

        if len(frames) == 0:
            return None
        return np.stack(frames)        # (T, 25, 3)

    except Exception:
        return None


def normalize_skeleton(joints: np.ndarray) -> np.ndarray:
    """
    Input:  (14, 3) — subset joints for one frame
    Output: (14, 3) — hip-centered, shoulder-width normalized

    This makes features invariant to:
      - person distance from camera (z-offset removed)
      - person height / body size (shoulder width scaling)
      - lateral position (x,y offset removed)
    """
    hip_center = (joints[HIP_L_IDX] + joints[HIP_R_IDX]) / 2.0
    joints_norm = joints - hip_center

    shoulder_dist = np.linalg.norm(
        joints_norm[SHOULDER_L_IDX] - joints_norm[SHOULDER_R_IDX]
    )
    if shoulder_dist > 0.05:           # skip degenerate frames
        joints_norm = joints_norm / shoulder_dist

    return joints_norm.astype(np.float32)


def extract_features(skeleton: np.ndarray) -> np.ndarray:
    """
    Input:  (T, 25, 3) full skeleton sequence
    Output: (T, FEATURE_DIM=84) feature sequence

    Feature layout per frame:
      [0 :42] normalized joint positions  (14 joints × 3)
      [42:84] joint velocities            (14 joints × 3)
    """
    T = skeleton.shape[0]

    # Subsample to 14 joints
    sub = skeleton[:, JOINT_SUBSET, :]  # (T, 14, 3)

    # Normalize each frame
    normed = np.stack([normalize_skeleton(sub[t]) for t in range(T)])  # (T, 14, 3)
    positions = normed.reshape(T, -1)   # (T, 42)

    # Velocities (finite differences, first frame = zero)
    velocities = np.zeros_like(positions)
    velocities[1:] = positions[1:] - positions[:-1]   # (T, 42)

    features = np.concatenate([positions, velocities], axis=-1)  # (T, 84)
    return features.astype(np.float32)


def skeleton_to_windows(
    skeleton: np.ndarray,
    label: int,
    window_size: int = WINDOW_SIZE,
    step: int = STEP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over a skeleton sequence and extract feature windows.

    Returns:
        X: (N_windows, window_size, FEATURE_DIM)
        y: (N_windows,)
    """
    features = extract_features(skeleton)     # (T, 84)
    T = features.shape[0]

    X_windows, y_windows = [], []

    for start in range(0, T - window_size + 1, step):
        window = features[start : start + window_size]   # (30, 84)
        X_windows.append(window)
        y_windows.append(label)

    if len(X_windows) == 0:
        return np.empty((0, window_size, FEATURE_DIM)), np.empty((0,), dtype=np.int64)

    return (
        np.stack(X_windows).astype(np.float32),
        np.array(y_windows, dtype=np.int64),
    )


def get_action_id(filename: str) -> int | None:
    """Extract action class from filename. e.g. 'S001C001P001R001A008' → 8"""
    try:
        base = os.path.splitext(os.path.basename(filename))[0]
        a_idx = base.index('A')
        return int(base[a_idx + 1 : a_idx + 4])
    except (ValueError, IndexError):
        return None


# ─── Main build function ──────────────────────────────────────────────────────

def build_dataset():
    print(f"\n{'='*60}")
    print("NTU RGB+D → Motion Dataset Builder")
    print(f"{'='*60}")
    print(f"Source : {NTU_DIR}")
    print(f"Output : {OUT_DIR}")
    print(f"Labels : {MOTION_LABELS}")
    print(f"Window : {WINDOW_SIZE} frames | Step: {STEP_SIZE} frames")
    print(f"Feature dim: {FEATURE_DIM} per frame\n")

    # Collect relevant files, grouped by label
    all_files = glob(os.path.join(NTU_DIR, "*.skeleton"))
    print(f"Total skeleton files found: {len(all_files)}")

    relevant = []
    skipped = 0
    for f in all_files:
        action_id = get_action_id(f)
        if action_id in NTU_TO_MOTION:
            relevant.append((f, NTU_TO_MOTION[action_id]))
        else:
            skipped += 1

    print(f"Relevant files (mapped labels): {len(relevant)}")
    print(f"Skipped (unmapped actions):     {skipped}\n")

    # Count per label before processing
    from collections import Counter
    label_counts = Counter(label for _, label in relevant)
    print("Files per label (before windowing):")
    for idx, name in MOTION_LABELS.items():
        print(f"  {idx} {name:<15}: {label_counts.get(idx, 0):>5} files")
    print()

    # Process all files
    X_all, y_all = [], []
    errors = 0
    t0 = time.time()

    for path, label in tqdm(relevant, desc="Parsing + windowing"):
        skeleton = parse_skeleton(path)

        if skeleton is None or len(skeleton) < MIN_FRAMES:
            errors += 1
            continue

        X_w, y_w = skeleton_to_windows(skeleton, label)
        if len(X_w) > 0:
            X_all.append(X_w)
            y_all.append(y_w)

    print(f"\nParsing done in {time.time()-t0:.1f}s  |  Errors/skipped: {errors}")

    if len(X_all) == 0:
        print("ERROR: No data collected. Check NTU_DIR path.")
        sys.exit(1)

    X = np.concatenate(X_all, axis=0)   # (N, 30, 84)
    y = np.concatenate(y_all, axis=0)   # (N,)

    print(f"\nTotal windows : {len(y):,}")
    print(f"Array shapes  : X={X.shape}  y={y.shape}")

    # Per-class count after windowing
    window_counts = Counter(y.tolist())
    print("\nWindows per label (after windowing):")
    for idx, name in MOTION_LABELS.items():
        print(f"  {idx} {name:<15}: {window_counts.get(idx, 0):>7,}")

    # Train / val split — 85% / 15%
    # IMPORTANT: shuffle before split so classes are distributed
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    val_size  = int(0.15 * len(y))
    X_val,   y_val   = X[:val_size],  y[:val_size]
    X_train, y_train = X[val_size:],  y[val_size:]

    print(f"\nTrain: {len(y_train):,}  |  Val: {len(y_val):,}")

    # Save
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(OUT_DIR, "y_val.npy"),   y_val)

    info = {
        "total_windows":  int(len(y)),
        "train_windows":  int(len(y_train)),
        "val_windows":    int(len(y_val)),
        "feature_dim":    FEATURE_DIM,
        "window_size":    WINDOW_SIZE,
        "num_classes":    NUM_CLASSES,
        "labels":         MOTION_LABELS,
        "class_counts_train": {
            MOTION_LABELS[k]: int(v)
            for k, v in Counter(y_train.tolist()).items()
        },
    }
    with open(os.path.join(OUT_DIR, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved to: {OUT_DIR}")
    print("  X_train.npy  y_train.npy")
    print("  X_val.npy    y_val.npy")
    print("  dataset_info.json")
    print("\nDone.")


if __name__ == "__main__":
    build_dataset()