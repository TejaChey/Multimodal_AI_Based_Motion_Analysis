"""
Quick test: Extract pose from video
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

# Paths
VIDEO_PATH = r"T:\Vlog Raws\sprint\recordings\Sachin\Sachin.mp4"
OUTPUT_PATH = r"results\test_output.mp4"
MODEL_PATH = r"Phase_1\data\pose_landmarker_heavy.task"

# MediaPipe Pose connections (standard 33 landmarks)
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
])

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    
    height, width, _ = annotated_image.shape

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        
        # Draw connections
        for connection in POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                start_lm = pose_landmarks[start_idx]
                end_lm = pose_landmarks[end_idx]
                
                # Check visibility/presence if needed (for now, assume all present)
                start_point = (int(start_lm.x * width), int(start_lm.y * height))
                end_point = (int(end_lm.x * width), int(end_lm.y * height))
                
                cv2.line(annotated_image, start_point, end_point, (245, 117, 66), 2)
                
        # Draw landmarks
        for landmark in pose_landmarks:
            point = (int(landmark.x * width), int(landmark.y * height))
            cv2.circle(annotated_image, point, 2, (245, 66, 230), -1)

    return annotated_image

def test_pose_extraction():
    """Test MediaPipe Tasks on one video"""
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f" Cannot open video: {VIDEO_PATH}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f" Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup PoseLandmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.VIDEO
    )
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    frame_num = 0
    landmarks_count = 0
    
    print("\n Processing video...")
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Calculate timestamp in ms
            timestamp_ms = int((frame_num / fps) * 1000)
            
            # Detect pose
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Draw skeleton if detected
            if detection_result.pose_landmarks:
                annotated_rgb = draw_landmarks_on_image(frame_rgb, detection_result)
                # Convert back to BGR for saving
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                out.write(annotated_bgr)
                landmarks_count += 1
            else:
                out.write(frame)
            
            frame_num += 1
            if frame_num % 30 == 0:
                print(f"  Processed {frame_num}/{total_frames} frames")
        
    cap.release()
    out.release()
    
    print(f"\n Done!")
    print(f" Detected pose in {landmarks_count}/{total_frames} frames ({100*landmarks_count/total_frames:.1f}%)")
    print(f" Output saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model missing! Run `wget -qO {MODEL_PATH} https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task` first.")
    else:
        test_pose_extraction()
