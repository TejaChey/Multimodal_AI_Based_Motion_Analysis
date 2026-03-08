# Sprint Analysis - Multimodal AI Project

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure
```
sprint_analysis/
├── data/utd-mhad/      # UTD-MHAD dataset (not in Git)
├── code/               # Processing scripts
├── results/            # Output features (not in Git)
└── venv/              # Virtual environment (not in Git)
```

## Usage
```bash
# Process videos
python code/process_all_videos.py

# Process IMU
python code/imu_feature_extractor.py

# Merge features
python code/merge_features.py
```

## Team
- Video Processing: [Teja]
- IMU Processing: [Shaik Sadiya]
