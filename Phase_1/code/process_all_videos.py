#!/usr/bin/env python3
"""Process ALL 32 jogging videos"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
from pathlib import Path

# Resolve paths relative to this script's location
_HERE = Path(__file__).resolve().parent        # sprint_analysis/code/
HOME = _HERE.parent                             # sprint_analysis/

RGB_PATH = HOME / "data/utd-mhad/RGB"
MODEL_PATH = HOME / "data/pose_landmarker_heavy.task"
OUTPUT_PATH = HOME / "results/video_features"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def calculate_angle(a, b, c):
    """Calculate angle at point b formed by points a-b-c"""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def process_one_video(video_path):
    """Extract features from one video using the MediaPipe Tasks API"""
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default if fps not detected

    features = []
    frame_num = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_num / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]  # First detected person

                # Get key landmarks
                l_shoulder = np.array([lm[11].x, lm[11].y])
                r_shoulder = np.array([lm[12].x, lm[12].y])
                l_hip = np.array([lm[23].x, lm[23].y])
                r_hip = np.array([lm[24].x, lm[24].y])
                l_knee = np.array([lm[25].x, lm[25].y])
                r_knee = np.array([lm[26].x, lm[26].y])
                l_ankle = np.array([lm[27].x, lm[27].y])
                r_ankle = np.array([lm[28].x, lm[28].y])

                # Calculate angles
                l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
                r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)

                # Hip center
                hip_center = (l_hip + r_hip) / 2

                features.append({
                    'left_knee_angle': l_knee_angle,
                    'right_knee_angle': r_knee_angle,
                    'left_hip_angle': l_hip_angle,
                    'right_hip_angle': r_hip_angle,
                    'hip_x': hip_center[0],
                    'hip_y': hip_center[1]
                })

            frame_num += 1

    cap.release()
    return pd.DataFrame(features)

def get_summary(df):
    """Calculate summary statistics"""
    if len(df) == 0:
        return None

    return {
        'knee_angle_mean': (df['left_knee_angle'].mean() + df['right_knee_angle'].mean()) / 2,
        'knee_angle_std': (df['left_knee_angle'].std() + df['right_knee_angle'].std()) / 2,
        'hip_angle_mean': (df['left_hip_angle'].mean() + df['right_hip_angle'].mean()) / 2,
        'hip_angle_std': (df['left_hip_angle'].std() + df['right_hip_angle'].std()) / 2,
        'knee_asymmetry': abs(df['left_knee_angle'].mean() - df['right_knee_angle'].mean()),
        'hip_asymmetry': abs(df['left_hip_angle'].mean() - df['right_hip_angle'].mean()),
        'frames_detected': len(df)
    }

def main():
    """Process all videos"""

    all_features = []

    print("Finding videos...")
    video_files = sorted(RGB_PATH.glob("a22_s*_t*_color.avi"))
    print(f"Found {len(video_files)} videos\n")

    for i, video_path in enumerate(video_files, 1):
        # Extract subject and trial from filename
        # a22_s1_t3_color.avi -> subject=1, trial=3
        parts = video_path.stem.split('_')
        subject = int(parts[1][1:])
        trial = int(parts[2][1:])

        print(f"[{i}/{len(video_files)}] Processing {video_path.name}...")

        # Process video
        df = process_one_video(video_path)

        if len(df) > 0:
            summary = get_summary(df)
            summary['subject'] = subject
            summary['trial'] = trial

            all_features.append(summary)
            print(f"     {len(df)} frames detected")
        else:
            print(f"     No pose detected")

    # Save results
    if all_features:
        results_df = pd.DataFrame(all_features)
        output_file = OUTPUT_PATH / "video_features.csv"
        results_df.to_csv(output_file, index=False)

        print(f"\n{'='*60}")
        print(f" SUCCESS!")
        print(f"{'='*60}")
        print(f"Processed: {len(results_df)} videos")
        print(f"Saved to: {output_file}")
        print(f"\n Sample data:")
        print(results_df.head())
        print(f"\n Statistics:")
        print(results_df[['knee_angle_mean', 'hip_angle_mean']].describe())
    else:
        print("\n No videos processed successfully!")

if __name__ == "__main__":
    main()
