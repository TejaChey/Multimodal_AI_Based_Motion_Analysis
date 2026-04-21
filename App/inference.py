import os
import sys
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

# Provide resolving path to Phase 2 processing scripts
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.append(str(PROJECT_ROOT)) 

from src.data.process_sprint_videos import process_video, get_sprint_metrics
from src.data.process_sprint_imu import process_file as process_imu

# ── MODEL DEPLOYMENT ARTIFACTS ───────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "Phase_3" / "results" / "model" / "best_sprint_model.keras"
SCALER_PATH = PROJECT_ROOT / "Phase_3" / "results" / "model" / "fitted_scaler.joblib"

# Load globally to avoid latency on every request
try:
    _model = tf.keras.models.load_model(str(MODEL_PATH))
    _scaler = joblib.load(str(SCALER_PATH))
except Exception as e:
    print(f"Warning: Model deployment files missing. {e}")

# Strict mathematical order dictated by Phase 3 training input shape
FEATURE_ORDER = [
    "duration_s", "total_frames", "knee_angle_mean", "hip_angle_mean",
    "knee_asymmetry", "hip_asymmetry", "vert_oscillation_norm",
    "total_steps_detected", "stride_freq_hz", "stride_len_m",
    "gct_ratio", "flight_ratio", "peak_accel_mag", "mean_accel_mag",
    "std_accel_mag", "stride_rate_accel_hz", "movement_smoothness_accel",
    "peak_gyro_mag", "mean_gyro_mag", "std_gyro_mag",
    "stride_rate_gyro_hz", "movement_smoothness_gyro", "steps_per_min_gyro"
]

def run_inference(video_path: Path, accel_path: Path, gyro_path: Path):
    """
    End-to-End Multimodal Inference Pipeline.
    Takes raw video + separate Galaxy Watch accelerometer & gyroscope CSVs,
    extracts 23 features, scales them, and predicts Skill Level.
    """
    # ── 1. Video Processing ──
    df_video, fps = process_video(video_path)
    if len(df_video) == 0:
        return None, None, "Video Error: Failed to extract human pose landmarks."
    
    video_metrics = get_sprint_metrics(df_video, fps)
    
    # ── 2. IMU Processing (merge accel + gyro) ──
    try:
        imu_metrics = process_imu(accel_path, gyro_path)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("\n" + "="*50)
        print("❗ IMU PROCESSING CRASH TRACEBACK:")
        print(tb)
        print("="*50 + "\n")
        return None, None, f"IMU Processing Error: {e}\n\nTraceback:\n{tb}"
        
    # Remove metadata keys we don't need for the neural network
    if 'athlete' in imu_metrics: del imu_metrics['athlete']
    if 'athlete' in video_metrics: del video_metrics['athlete']
    
    # Merge dicts
    combined = {**video_metrics, **imu_metrics}
    
    # ── 3. Normalization & Prediction ──
    vector = []
    for feature in FEATURE_ORDER:
        vector.append(float(combined.get(feature, 0.0)))
        
    X = np.array(vector).reshape(1, -1)
    
    try:
        X_scaled = _scaler.transform(X)
        preds = _model.predict(X_scaled, verbose=0)[0]
    except Exception as e:
        return None, None, f"Inference Math Error: Cannot evaluate features. {e}"
        
    class_idx = np.argmax(preds)
    confidence = float(preds[class_idx])
    
    classes = {0: "Beginner", 1: "Intermediate", 2: "Elite"}
    result_label = classes[class_idx]
    
    return result_label, combined, ""
