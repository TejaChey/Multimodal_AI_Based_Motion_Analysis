# Multimodal AI-Based Motion Analysis

A multimodal AI pipeline for human motion analysis using video (MediaPipe pose estimation) and IMU sensor data from the [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) dataset.

---

## 📁 Project Structure

```
Multimodal-AI-Based-Motion-Analysis-/
├── Phase_1/
│   ├── code/                         # All processing and analysis scripts
│   │   ├── process_all_videos.py     # Extract pose landmarks from all videos
│   │   ├── imu_feature_extractoripynb.py  # Extract features from IMU data
│   │   ├── merge_features.py         # Merge video + IMU features
│   │   ├── explore_data.py           # Data exploration and statistics
│   │   ├── prepare_ml_data.py        # Prepare ML datasets (binary)
│   │   ├── prepare_ml_data_3class.py # Prepare ML datasets (3-class)
│   │   ├── train_model.py            # Train classification model
│   │   ├── compare_modalities.py     # Compare modalities (binary)
│   │   ├── compare_modalities_3class.py    # Compare modalities (3-class)
│   │   ├── compare_modalities_improved.py  # Improved modality comparison
│   │   └── test_mediapipe.py         # MediaPipe setup test
│   ├── data/
│   │   └── utd-mhad/                 # UTD-MHAD dataset (not in Git)
│   └── results/                      # Generated outputs (not in Git)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# Clone the repository
git clone <repo-url>
cd Multimodal-AI-Based-Motion-Analysis-

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset
Download the [UTD-MHAD dataset](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) and place it at:
```
Phase_1/data/utd-mhad/
```

Also download the MediaPipe Pose Landmarker model:
```
Phase_1/data/pose_landmarker_heavy.task
```

---

## 🚀 Usage

Run scripts from the project root with the virtual environment activated:

```bash
# 1. Extract pose features from video data
python Phase_1/code/process_all_videos.py

# 2. Extract IMU features
python Phase_1/code/imu_feature_extractoripynb.py

# 3. Merge video + IMU features
python Phase_1/code/merge_features.py

# 4. Explore the data
python Phase_1/code/explore_data.py

# 5. Prepare ML datasets
python Phase_1/code/prepare_ml_data_3class.py

# 6. Train model
python Phase_1/code/train_model.py

# 7. Compare modalities
python Phase_1/code/compare_modalities_improved.py
```

---

## 🧪 Phase 1: Modality Comparison

Phase 1 benchmarks three modalities for action classification:
- **Video only** – MediaPipe pose landmarks
- **IMU only** – Raw sensor feature extraction
- **Multimodal** – Fusion of video + IMU features

Classification is performed using Random Forest and SVM classifiers across binary and 3-class setups.

---

## 👥 Team

| Role | Member |
|------|--------|
| Video Processing | Teja |
| IMU Processing | Shaik Sadiya |
