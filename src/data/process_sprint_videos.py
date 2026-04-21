#!/usr/bin/env python3
"""
Phase 2 Sprint Video Processor
------------------------------
Extracts 60fps full-body sprint biomechanics from 15m fly zone videos using MediaPipe Pose.
Calculates classical angles PLUS sprint-specific metrics: Stride Length, Stride Frequency,
Ground Contact Time (GCT), Flight Time, and Vertical Oscillation.

Run: python Phase_2/code/process_sprint_videos.py
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Classic CPU-based MediaPipe Pose (compatible with headless cloud servers)
_mp_pose = mp.solutions.pose


# ── CONFIG ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "Phase_2" / "data" / "sprint_raw"
MODEL_PATH   = PROJECT_ROOT / "Phase_1" / "data" / "pose_landmarker_heavy.task"
OUTPUT_DIR   = PROJECT_ROOT / "Phase_2" / "results" / "video_features"
OUTPUT_CSV   = OUTPUT_DIR / "video_features.csv"

# Known sprint fly-zone distance
FLY_ZONE_METERS = 15.0

# ── HELPERS ───────────────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    """Calculate 2D angle at joint b (degrees)"""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def detect_strides(ankle_y_coords, fps):
    """
    Detect strides by finding peaks in the ankle's Y trajectory.
    In OpenCV, Y increases downwards, so the ankle hits the ground at MAX Y.
    """
    # Minimum distance between steps (e.g., at 60fps, 3 steps/sec = 20 frames)
    min_dist = int(fps * 0.25)
    
    peaks, _ = find_peaks(ankle_y_coords, distance=min_dist, prominence=0.02)
    return len(peaks), peaks

# ── MAIN PROCESSOR ────────────────────────────────────────────────────────────

def process_video(video_path: Path) -> pd.DataFrame:
    """Extracts frame-by-frame landmarks into a DataFrame using CPU-compatible MediaPipe."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"   Cannot open {video_path.name}")
        return pd.DataFrame(), 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frames_data = []
    frame_idx = 0

    with _mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            ts_ms = int((frame_idx / fps) * 1000)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark

                l_shoulder = np.array([lm[11].x, lm[11].y])
                r_shoulder = np.array([lm[12].x, lm[12].y])
                l_hip      = np.array([lm[23].x, lm[23].y])
                r_hip      = np.array([lm[24].x, lm[24].y])
                l_knee     = np.array([lm[25].x, lm[25].y])
                r_knee     = np.array([lm[26].x, lm[26].y])
                l_ankle    = np.array([lm[27].x, lm[27].y])
                r_ankle    = np.array([lm[28].x, lm[28].y])

                frames_data.append({
                    'time_ms': ts_ms,
                    'l_knee_angle': calculate_angle(l_hip, l_knee, l_ankle),
                    'r_knee_angle': calculate_angle(r_hip, r_knee, r_ankle),
                    'l_hip_angle':  calculate_angle(l_shoulder, l_hip, l_knee),
                    'r_hip_angle':  calculate_angle(r_shoulder, r_hip, r_knee),
                    'hip_y':        (l_hip[1] + r_hip[1]) / 2.0,
                    'athlete_height_px': abs(l_shoulder[1] - l_ankle[1]),
                    'l_ankle_y':    l_ankle[1],
                    'r_ankle_y':    r_ankle[1]
                })

            frame_idx += 1

    cap.release()
    return pd.DataFrame(frames_data), fps



def get_sprint_metrics(df: pd.DataFrame, fps: float) -> dict:
    """Computes advanced aggregated biomechanics from the full time series."""
    if len(df) < 10:
        return None

    duration_s = (df['time_ms'].max() - df['time_ms'].min()) / 1000.0
    if duration_s <= 0:
        duration_s = len(df) / fps

    # 1. Angles & Symmetry
    l_knee_m = df['l_knee_angle'].mean()
    r_knee_m = df['r_knee_angle'].mean()
    l_hip_m  = df['l_hip_angle'].mean()
    r_hip_m  = df['r_hip_angle'].mean()
    
    knee_asym = abs(l_knee_m - r_knee_m)
    hip_asym  = abs(l_hip_m - r_hip_m)

    # 2. Vertical Oscillation (Hip bounce normalized by athlete height)
    # Using std deviation of hip Y relative to their body size
    mean_height = df['athlete_height_px'].mean()
    if mean_height > 0:
        vert_oscillation = df['hip_y'].std() / mean_height
    else:
        vert_oscillation = df['hip_y'].std()

    # 3. Strides (detecting foot plants)
    l_strides, _ = detect_strides(df['l_ankle_y'].values, fps)
    r_strides, _ = detect_strides(df['r_ankle_y'].values, fps)
    total_steps = l_strides + r_strides
    
    # Safeties for very short videos
    if total_steps < 2:
        stride_freq = 0.0
        stride_len  = 0.0
    else:
        stride_freq = total_steps / duration_s
        # Total distance covered = 15m. Each step covers a portion. 
        # Actually total_steps = 15m / (avg stride length)
        # So stride_len = 15m / total_steps 
        # (Assuming video perfectly brackets the fly zone)
        stride_len = FLY_ZONE_METERS / total_steps

    # 4. GCT & Flight Time (Proxy)
    # Ground contact happens when vertical velocity of ankle is ~0
    # Calculate difference in Y over time
    l_vy = np.abs(np.diff(df['l_ankle_y'].values))
    r_vy = np.abs(np.diff(df['r_ankle_y'].values))
    
    # Threshold for "stillness" (normalized units per frame)
    gct_threshold = 0.005
    
    l_gct_frames = np.sum(l_vy < gct_threshold)
    r_gct_frames = np.sum(r_vy < gct_threshold)
    
    total_frames = len(df) - 1 # diff array is 1 shorter
    gct_ratio = (l_gct_frames + r_gct_frames) / (2.0 * max(1, total_frames))
    flight_ratio = 1.0 - gct_ratio

    return {
        'duration_s': round(duration_s, 3),
        'total_frames': len(df),
        'knee_angle_mean': round((l_knee_m + r_knee_m) / 2, 2),
        'hip_angle_mean': round((l_hip_m + r_hip_m) / 2, 2),
        'knee_asymmetry': round(knee_asym, 2),
        'hip_asymmetry': round(hip_asym, 2),
        'vert_oscillation_norm': round(vert_oscillation, 5),
        'total_steps_detected': total_steps,
        'stride_freq_hz': round(stride_freq, 2),
        'stride_len_m': round(stride_len, 2),
        'gct_ratio': round(gct_ratio, 3),
        'flight_ratio': round(flight_ratio, 3)
    }

# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'='*50}\n  🏃 Phase 2: Sprint Video Processor\n{'='*50}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MediaPipe Model missing at {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Iterate dynamically through all athlete folders
    athlete_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir()])
    
    for folder in athlete_dirs:
        # Find video in this folder
        videos = [f for f in folder.iterdir() if f.suffix.lower() in {'.mp4', '.avi', '.mov'}]
        if not videos:
            continue
            
        video_path = videos[0]
        athlete_name = folder.name
        
        print(f"[{len(results)+1}/{len(athlete_dirs)}] Processing {athlete_name}... ", end="", flush=True)
        
        try:
            df_frames, fps = process_video(video_path)
            
            if len(df_frames) > 0:
                metrics = get_sprint_metrics(df_frames, fps)
                if metrics:
                    metrics['athlete'] = athlete_name
                    # Push athlete name to front
                    metrics = {'athlete': metrics.pop('athlete'), **metrics}
                    results.append(metrics)
                    print(f" {metrics['total_steps_detected']} steps | {metrics['stride_freq_hz']} Hz")
                else:
                    print(" (Too few frames)")
            else:
                print(" (No pose detected)")
        except Exception as e:
            print(f" (Error: {e})")

    # Save
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\n All done! Processed {len(df_out)} videos.")
        print(f" Saved to: {OUTPUT_CSV}")
        print("\nPreview:")
        print(df_out[['athlete', 'stride_freq_hz', 'stride_len_m', 'vert_oscillation_norm']].head())
    else:
        print("\n No features extracted. Check video paths and MediaPipe model.")
