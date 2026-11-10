"""Video decoding and MediaPipe Pose extraction."""
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def video_meta(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    meta = {"fps": float(cap.get(cv2.CAP_PROP_FPS)), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "creation_utc": None}
    cap.release()
    try:  # optional: container creation time (used for the clock-consistency check)
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
                              "-of", "json", str(path)], capture_output=True, text=True, timeout=20).stdout
        meta["creation_utc"] = json.loads(out)["format"]["tags"]["creation_time"]
    except Exception:
        pass
    return meta


def extract_pose(path, p: dict) -> pd.DataFrame:
    """All 33 landmarks per frame, in pixels. Adds per-frame pose latency."""
    from mediapipe.python.solutions import pose as mp_pose  # lazy: heavy import
    cfg = p["pose"]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    rows = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=cfg["complexity"],
                      min_detection_confidence=cfg["min_detection_confidence"],
                      min_tracking_confidence=cfg["min_tracking_confidence"]) as pose:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            row = {"frame": i, "t": i / fps, "pose_ms": (time.perf_counter() - t0) * 1e3,
                   "detected": res.pose_landmarks is not None, "W": W, "H": H, "fps": fps}
            if res.pose_landmarks:
                for k, lm in enumerate(res.pose_landmarks.landmark):
                    row[f"x{k}"], row[f"y{k}"] = lm.x * W, lm.y * H
                    row[f"z{k}"], row[f"v{k}"] = lm.z, lm.visibility
            rows.append(row)
            i += 1
    cap.release()
    return pd.DataFrame(rows)


def runner_track(df: pd.DataFrame, max_gap=6, max_jump_px=60, size_ratio=1.5) -> pd.DataFrame:
    """Keep only frames of the sprinting athlete.

    MediaPipe can lock onto a static false detection (stands, bystander) before the
    runner enters the frame. Split detections into continuous tracks (no long gaps,
    no position jumps, stable body size) and keep the one with the largest travel.
    """
    d = df[df.detected].copy()
    if len(d) < 2:
        return d
    hx = ((d.x23 + d.x24) / 2).values
    size = np.hypot(d.x11 - d.x23, d.y11 - d.y23).values
    fr = d.frame.values
    seg, cur = [0], 0
    for i in range(1, len(d)):
        gap = fr[i] - fr[i - 1]
        ratio = max(size[i], size[i - 1]) / max(min(size[i], size[i - 1]), 1e-6)
        if gap > max_gap or abs(hx[i] - hx[i - 1]) / gap > max_jump_px or ratio > size_ratio:
            cur += 1
        seg.append(cur)
    seg = np.array(seg)
    travel = {s: np.ptp(hx[seg == s]) for s in np.unique(seg)}
    return d[seg == max(travel, key=travel.get)]
