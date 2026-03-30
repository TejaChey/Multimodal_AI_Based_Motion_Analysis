#!/usr/bin/env python3
"""Merge video + IMU features"""

import pandas as pd
from pathlib import Path

_HERE     = Path(__file__).resolve().parent.parent  # Phase_1/
VIDEO_CSV = _HERE / "results" / "video_features" / "video_features.csv"
IMU_CSV   = _HERE / "results" / "imu_features" / "imu_features.csv"
OUTPUT_CSV = _HERE / "results" /  "multimodal_features" /"multimodal_features.csv"

def merge():
    print("Loading features...")
    
    # Load
    video_df = pd.read_csv(VIDEO_CSV)
    imu_df = pd.read_csv(IMU_CSV)
    
    print(f"  Video: {len(video_df)} samples, {len(video_df.columns)} features")
    print(f"  IMU: {len(imu_df)} samples, {len(imu_df.columns)} features")
    
    # Merge on subject and trial
    merged = pd.merge(
        video_df, 
        imu_df, 
        on=['subject', 'trial'], 
        how='inner'
    )
    
    print(f"\nMerged: {len(merged)} samples")
    print(f"Total features: {len(merged.columns)}")
    
    # Save
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f" Saved: {OUTPUT_CSV}")
    
    print(f"\n Feature columns:")
    print(merged.columns.tolist())
    
    print(f"\n Preview:")
    print(merged.head())
    
    return merged

if __name__ == "__main__":
    merge()