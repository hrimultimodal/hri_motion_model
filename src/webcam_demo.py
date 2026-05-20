"""
webcam_demo.py

Real-time motion classification using laptop webcam.
Uses MediaPipe BlazePose for skeleton extraction.
Displays live skeleton overlay + motion prediction on screen.

Controls:
    Q or ESC  → quit
    R         → reset buffer (use when person re-enters frame)
    S         → save current frame as screenshot

Usage:
    python webcam_demo.py
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from inference import MotionInference, MOTION_LABELS, NUM_CLASSES

# ─── Paths ────────────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
CKPT     = os.path.join(SRC_DIR, "..", "checkpoints", "best_model.pt")

# ─── Colours (BGR) ────────────────────────────────────────────────────────────
COLOURS = {
    "sitting":      (  0, 200, 255),   # yellow
    "standing":     (  0, 255,   0),   # green
    "walking":      (255, 200,   0),   # cyan
    "running":      (  0, 100, 255),   # orange
    "stepping_back":(  0,   0, 255),   # red
    "slumped":      (255,   0, 200),   # pink
    "buffering":    (180, 180, 180),   # grey
}

# Skeleton connections to draw (pairs of MediaPipe landmark indices)
SKELETON_CONNECTIONS = [
    (11, 12),             # shoulders
    (11, 13), (13, 15),   # left arm
    (12, 14), (14, 16),   # right arm
    (11, 23), (12, 24),   # torso sides
    (23, 24),             # hips
    (23, 25), (25, 27),   # left leg
    (24, 26), (26, 28),   # right leg
]


def draw_skeleton(frame, landmarks, colour, h, w):
    """Draw skeleton lines and joint dots on frame."""
    # Draw connections
    for a, b in SKELETON_CONNECTIONS:
        lm_a = landmarks[a]
        lm_b = landmarks[b]
        if lm_a.visibility > 0.4 and lm_b.visibility > 0.4:
            pt_a = (int(lm_a.x * w), int(lm_a.y * h))
            pt_b = (int(lm_b.x * w), int(lm_b.y * h))
            cv2.line(frame, pt_a, pt_b, colour, 2, cv2.LINE_AA)

    # Draw joint dots
    for idx in range(33):
        lm = landmarks[idx]
        if lm.visibility > 0.4:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 4, colour, -1, cv2.LINE_AA)


def draw_prob_bars(frame, probs, current_label, x=10, y=200):
    """Draw a small probability bar chart in the corner."""
    bar_w   = 120
    bar_h   = 14
    padding = 4

    for i, (idx, name) in enumerate(MOTION_LABELS.items()):
        p      = float(probs[idx]) if len(probs) == NUM_CLASSES else 0.0
        colour = COLOURS.get(name, (200, 200, 200))

        # Background bar
        y0 = y + i * (bar_h + padding)
        cv2.rectangle(frame, (x, y0), (x + bar_w, y0 + bar_h),
                      (50, 50, 50), -1)

        # Filled bar
        filled = int(p * bar_w)
        cv2.rectangle(frame, (x, y0), (x + filled, y0 + bar_h),
                      colour, -1)

        # Label + probability text
        bold = cv2.FONT_HERSHEY_SIMPLEX
        label_str = f"{name[:12]:<12} {p:.2f}"
        cv2.putText(frame, label_str, (x + bar_w + 6, y0 + bar_h - 2),
                    bold, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def draw_prediction(frame, result, h, w):
    """Draw the main prediction label at the top of the frame."""
    label  = result.label
    conf   = result.confidence
    colour = COLOURS.get(label, (200, 200, 200))

    # Background rectangle
    text    = f"{label.upper()}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    margin  = 10
    cv2.rectangle(frame,
                  (w // 2 - tw // 2 - margin, 8),
                  (w // 2 + tw // 2 + margin, 50),
                  (30, 30, 30), -1)

    # Text
    cv2.putText(frame, text,
                (w // 2 - tw // 2, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, colour, 2, cv2.LINE_AA)

    # Buffering notice
    if not result.stable:
        cv2.putText(frame, f"Buffering... {len(result.probs)}/30",
                    (w // 2 - 80, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


def main():
    # ── Load inference engine ─────────────────────────────────────────────────
    print("Loading model...")
    engine = MotionInference(CKPT)

    # ── MediaPipe pose ────────────────────────────────────────────────────────
    mp_pose    = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,        # 0=Lite, 1=Full — laptop can handle Full
        enable_segmentation=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        static_image_mode=False,
    )

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Try changing VideoCapture(0) to (1).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    print("\nWebcam demo running.")
    print("Controls:  Q/ESC = quit   |   R = reset buffer   |   S = screenshot\n")

    # FPS tracking
    fps_buffer = deque(maxlen=30)
    prev_time  = time.time()
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed.")
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        # ── Pose estimation ───────────────────────────────────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        person_detected = results.pose_landmarks is not None

        if person_detected:
            landmarks = results.pose_landmarks.landmark

            # Build (25, 3) array from MediaPipe world landmarks
            # MediaPipe has 33 landmarks — we use indices 0-24 which
            # roughly correspond to the NTU joint layout
            world_lms = results.pose_world_landmarks.landmark
            joints_25 = np.zeros((25, 3), dtype=np.float32)
            # Map MediaPipe → NTU joint indices (approximate)
            mp_to_ntu = {
                0: 3,   # nose        → head
                11: 4,  # l_shoulder  → left_shoulder
                12: 8,  # r_shoulder  → right_shoulder
                13: 5,  # l_elbow     → left_elbow
                14: 9,  # r_elbow     → right_elbow
                15: 6,  # l_wrist     → left_wrist
                16: 10, # r_wrist     → right_wrist
                23: 12, # l_hip       → left_hip
                24: 16, # r_hip       → right_hip
                25: 13, # l_knee      → left_knee
                26: 17, # r_knee      → right_knee
                27: 14, # l_ankle     → left_ankle
                28: 18, # r_ankle     → right_ankle
            }
            for mp_idx, ntu_idx in mp_to_ntu.items():
                lm = world_lms[mp_idx]
                joints_25[ntu_idx] = [lm.x, lm.y, lm.z]

            # Approximate spine joints from shoulder/hip midpoints
            joints_25[0] = (joints_25[12] + joints_25[16]) / 2  # base_spine
            joints_25[1] = (joints_25[4]  + joints_25[8])  / 2  # mid_spine
            joints_25[2] = joints_25[1] * 0.5 + joints_25[3] * 0.5  # neck

            # Feed to inference engine
            result = engine.update(joints_25)

            # Draw skeleton in prediction colour
            colour = COLOURS.get(result.label, (200, 200, 200))
            draw_skeleton(frame, landmarks, colour, h, w)

        else:
            # No person — feed zeros to keep buffer advancing
            # but reset if absent for >1 second
            result = engine.update(np.zeros((25, 3), dtype=np.float32))

        # ── FPS ───────────────────────────────────────────────────────────────
        now = time.time()
        fps_buffer.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = np.mean(fps_buffer)

        # ── Draw UI ───────────────────────────────────────────────────────────
        # Dark overlay strip at bottom for prob bars
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 145), (270, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        draw_prediction(frame, result, h, w)
        draw_prob_bars(frame, result.probs, result.label,
                       x=10, y=h - 140)
        draw_fps(frame, fps)

        # No person indicator
        if not person_detected:
            cv2.putText(frame, "No person detected", (w // 2 - 100, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 1)

        cv2.imshow("HRI Motion Detector", frame)

        # ── Key controls ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):      # Q or ESC
            break
        elif key == ord('r'):
            engine.reset()
            print("Buffer reset.")
        elif key == ord('s'):
            fname = f"screenshot_{frame_idx:04d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Done.")


if __name__ == "__main__":
    main()