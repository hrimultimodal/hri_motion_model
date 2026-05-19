"""
explore_skeleton.py
Reads ONE .skeleton file and prints its structure clearly.
Run this to verify we understand the format before writing the real parser.
"""

import sys
import numpy as np

# NTU RGB+D 25 joint names (index → name)
JOINT_NAMES = [
    "base_of_spine",       #  0
    "mid_spine",           #  1
    "neck",                #  2
    "head",                #  3
    "left_shoulder",       #  4
    "left_elbow",          #  5
    "left_wrist",          #  6
    "left_hand",           #  7
    "right_shoulder",      #  8
    "right_elbow",         #  9
    "right_wrist",         # 10
    "right_hand",          # 11
    "left_hip",            # 12
    "left_knee",           # 13
    "left_ankle",          # 14
    "left_foot",           # 15
    "right_hip",           # 16
    "right_knee",          # 17
    "right_ankle",         # 18
    "right_foot",          # 19
    "spine",               # 20
    "tip_of_left_hand",    # 21
    "left_thumb",          # 22
    "tip_of_right_hand",   # 23
    "right_thumb",         # 24
]


def read_skeleton_file(path: str):
    """
    Returns a list of frames.
    Each frame is a numpy array of shape (25, 3) — 25 joints, xyz in meters.
    Returns None if file is empty or malformed.
    """
    with open(path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    idx = 0
    total_frames = int(lines[idx]); idx += 1
    frames = []

    for frame_idx in range(total_frames):
        num_bodies = int(lines[idx]); idx += 1

        if num_bodies == 0:
            # No person in this frame — store zeros, keep frame count consistent
            frames.append(np.zeros((25, 3), dtype=np.float32))
            continue

        # Read first body only (we handle single-person scenarios)
        idx += 1  # skip body metadata line
        num_joints = int(lines[idx]); idx += 1

        joints = np.zeros((25, 3), dtype=np.float32)
        for j in range(num_joints):
            vals = lines[idx].split()
            joints[j, 0] = float(vals[0])  # x
            joints[j, 1] = float(vals[1])  # y
            joints[j, 2] = float(vals[2])  # z
            idx += 1

        frames.append(joints)

        # Skip any additional bodies in this frame
        for _ in range(num_bodies - 1):
            idx += 1   # body metadata
            nj = int(lines[idx]); idx += 1
            idx += nj  # skip joints

    return np.stack(frames)  # shape: (T, 25, 3)


def explore(path: str):
    print(f"\n{'='*60}")
    print(f"File: {path.split('/')[-1]}")
    print(f"{'='*60}")

    skeleton = read_skeleton_file(path)
    T = skeleton.shape[0]

    print(f"\nTotal frames : {T}")
    print(f"Array shape  : {skeleton.shape}  (frames × joints × xyz)")
    print(f"Duration     : ~{T/30:.1f} seconds  (assuming 30 FPS)")

    print(f"\n--- Frame 0, all 25 joints ---")
    print(f"{'Idx':<4} {'Joint Name':<22} {'X (m)':>8} {'Y (m)':>8} {'Z (m)':>8}")
    print("-" * 55)
    for j, name in enumerate(JOINT_NAMES):
        x, y, z = skeleton[0, j]
        print(f"{j:<4} {name:<22} {x:>8.4f} {y:>8.4f} {z:>8.4f}")

    print(f"\n--- Z-depth range across all frames (distance from camera) ---")
    # Joint 0 = base of spine = body center
    spine_z = skeleton[:, 0, 2]
    print(f"  Spine Z: min={spine_z.min():.2f}m  max={spine_z.max():.2f}m  "
          f"mean={spine_z.mean():.2f}m")

    print(f"\n--- Hip Y-position (proxy for sitting vs standing) ---")
    left_hip_y  = skeleton[:, 12, 1]
    right_hip_y = skeleton[:, 16, 1]
    hip_y = (left_hip_y + right_hip_y) / 2
    print(f"  Hip Y: min={hip_y.min():.3f}  max={hip_y.max():.3f}  "
          f"mean={hip_y.mean():.3f}")

    print(f"\n--- Wrist height (proxy for raised hand gesture) ---")
    left_wrist_y  = skeleton[:, 6, 1]
    right_wrist_y = skeleton[:, 10, 1]
    neck_y = skeleton[:, 2, 1]
    print(f"  Left wrist Y  : mean={left_wrist_y.mean():.3f}")
    print(f"  Right wrist Y : mean={right_wrist_y.mean():.3f}")
    print(f"  Neck Y        : mean={neck_y.mean():.3f}")

    print(f"\n--- Body velocity (motion intensity) ---")
    # How much does the spine base move per frame?
    spine_pos = skeleton[:, 0, :]   # (T, 3)
    displacements = np.linalg.norm(np.diff(spine_pos, axis=0), axis=1)
    print(f"  Per-frame displacement: mean={displacements.mean():.4f}m  "
          f"max={displacements.max():.4f}m")
    print(f"  Estimated speed: {displacements.mean()*30:.3f} m/s  "
          f"(peak: {displacements.max()*30:.3f} m/s)")


if __name__ == "__main__":
    import os

    ntu_dir = os.environ.get("NTU_DIR", ".")

    # Explore a few different action classes to see the contrast
    test_files = [
        "S001C001P001R001A001.skeleton",   # A001 = drinking water (standing)
        "S001C001P001R001A008.skeleton",   # A008 = sitting down
        "S001C001P001R001A010.skeleton",   # A010 = clapping (standing)
    ]

    for fname in test_files:
        fpath = os.path.join(ntu_dir, fname)
        if os.path.exists(fpath):
            explore(fpath)
        else:
            print(f"Not found: {fpath}")