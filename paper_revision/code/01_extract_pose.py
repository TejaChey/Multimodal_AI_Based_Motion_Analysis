"""Run MediaPipe Pose on every sprint clip and cache all 33 landmarks per frame.

Also records per-frame wall-clock pose latency, reused by the latency benchmark.
Output: results/pose/<clip>.csv
"""
import time

import cv2
import numpy as np
import pandas as pd
from mediapipe.python.solutions import pose as mp_pose

from common import CLIPS, POSE_DIR, video_path

COMPLEXITY = 1  # same setting as the deployed app (App/inference.py)


def extract(clip: str) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path(clip)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    rows = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=COMPLEXITY,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dt = (time.perf_counter() - t0) * 1000
            row = {"frame": i, "t": i / fps, "pose_ms": dt, "detected": res.pose_landmarks is not None,
                   "W": W, "H": H, "fps": fps}
            if res.pose_landmarks:
                for k, lm in enumerate(res.pose_landmarks.landmark):
                    # store in pixels so x and y share one scale
                    row[f"x{k}"], row[f"y{k}"] = lm.x * W, lm.y * H
                    row[f"z{k}"], row[f"v{k}"] = lm.z, lm.visibility
            rows.append(row)
            i += 1
    cap.release()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for clip in CLIPS:
        df = extract(clip)
        df.to_csv(POSE_DIR / f"{clip}.csv", index=False)
        print(f"{clip:18s} frames={len(df):4d} detected={df.detected.mean():.0%} "
              f"pose_ms median={df.pose_ms.median():.1f}")
