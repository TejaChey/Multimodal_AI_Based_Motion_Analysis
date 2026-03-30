import pandas as pd
from pathlib import Path

# Paths
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent

VIDEO_FEATURES_CSV = PROJECT_ROOT / "Phase_2" / "results" / "video_features" / "video_features.csv"
IMU_FEATURES_CSV = PROJECT_ROOT / "Phase_2" / "results" / "imu_features" / "imu_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "Phase_2" / "results"
OUTPUT_CSV = OUTPUT_DIR / "multimodal_features.csv"

def main():
    print("Starting feature fusion process...")
    
    # Check if necessary files exist
    if not VIDEO_FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing video features at {VIDEO_FEATURES_CSV}")
    if not IMU_FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing IMU features at {IMU_FEATURES_CSV}")
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load the datasets
    df_video = pd.read_csv(VIDEO_FEATURES_CSV)
    df_imu = pd.read_csv(IMU_FEATURES_CSV)
    
    print(f"Loaded {len(df_video)} video features and {len(df_imu)} IMU features.")
    
    # Merge datasets based on the athlete name column using an inner join
    # This prevents any misaligned data from entering the final dataset
    df_merged = pd.merge(df_video, df_imu, on='athlete', how='inner')
    
    # Move the athlete column to the very front for readability
    if 'athlete' in df_merged.columns:
        cols = ['athlete'] + [col for col in df_merged.columns if col != 'athlete']
        df_merged = df_merged[cols]
        
    # Save the final fused dataset
    df_merged.to_csv(OUTPUT_CSV, index=False)
    
    print("Merge complete.")
    print(f"Final feature count per athlete: {len(df_merged.columns) - 1}")
    print(f"Total athletes successfully merged: {len(df_merged)}")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
