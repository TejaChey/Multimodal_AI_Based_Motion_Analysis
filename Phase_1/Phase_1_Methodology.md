# Phase 1: Multimodal Jogging Biomechanics Methodology

## 1. Project Objective
The primary objective of Phase 1 was to establish a foundational **Multimodal Machine Learning Pipeline** capable of synchronizing and analyzing human motion through two distinct data streams: optical video and wearable inertial sensors. Using this pipeline, we aimed to classify the intensity of a repetitive human action (Jogging) into a binary classification problem: **Slow vs. Fast**.

## 2. Dataset Overview
We utilized the **UTD-MHAD (University of Texas at Dallas Multimodal Human Action Dataset)**. It is a benchmark academic dataset containing synchronized multimodal data of subjects performing various actions. For Phase 1, we isolated the **Jogging in Place (Action #21)** sequences.

**Data Streams Utilized:**
1. **RGB Video**: Standard 30fps optical camera footage capturing the full body of the subject.
2. **Wearable IMU**: A sensor worn on the right wrist/thigh capturing tri-axial Accelerometer and Gyroscope data at high sampling frequencies.

## 3. Data Processing Architecture

The pipeline uses a **Late Fusion** architecture. Instead of merging the raw sensor time-series with video pixels directly, we independently extracted engineered biomechanical features from both modalities before fusing them mathematically.

### A. Vision Processing (`process_all_videos.py`)
To process the visual data without relying on manual landmarking or physical mocap suits, we leveraged **Google's MediaPipe Pose**. 
1. **PoseLandmarker Engine**: We initialized the `PoseLandmarker` in `VIDEO` mode to track 33 3D anatomical joints frame-by-frame.
2. **Feature Extraction**: As the video processed, we continuously calculated the physical displacement (velocity) of key joints (ankles, knees, hips).
3. **Aggregation**: The time-series 3D coordinates were compressed into aggregate features per video, such as:
   - Mean knee velocity
   - Mean vertical ankle displacement
   - Estimated step frequency

### B. IMU Processing (`process_all_imu.py`)
The inertial data provided high-fidelity ground truth for physical exertion that the camera cannot visually see (e.g., G-force impacts).
1. **Magnitude Calculation**: We computed the absolute magnitude of the 3D acceleration vector: $Mag = \sqrt{X^2 + Y^2 + Z^2}$.
2. **Feature Extraction**: We extracted statistical features outlining the intensity and rhythm of the jog:
   - Peak Acceleration (Impact force)
   - Mean Gyroscope variance (Rotational movement)
   - Stride consistency calculations

### C. Feature Merging (`prepare_ml_data.py`)
1. **Temporal Alignment**: Because the camera operates at 30Hz and the IMU operates at ~50+Hz, the timelines were mathematically aligned and trimmed to evaluate only the active "jogging" window.
2. **Concatenation**: Every subject outputted a single flat vector combining the Visual Features and IMU Features into a single 1D array.

## 4. Machine Learning Classification (`train_model.py`)

With the multimodal features combined into a unified dataset, we passed them into a classical supervised Machine Learning pipeline.

1. **Pre-Processing**: The features underwent Z-score normalization using `sklearn.preprocessing.StandardScaler` to ensure that large magnitude variances (like G-forces) didn't overshadow small value changes (like bounding box velocities).
2. **Dataset Splitting**: The data was split into `X_train` and `X_test` datasets using an 80/20 train-test split.
3. **Model Selection**: We trained two separate classification models to compare architectures:
   - **Random Forest Classifier**: An ensemble learning method composed of 100 decision trees. Excellent for non-linear biomechanical data.
   - **Support Vector Machine (SVM)**: Outfitted with an RBF (Radial Basis Function) kernel to map the data into higher-dimensional spaces.

## 5. Evaluation & Output
The models predicted whether a subject was jogging **Slow** or **Fast**. The pipeline produced:
- **Accuracy Scores**: To directly compare the Random Forest against the SVM.
- **Confusion Matrix**: A heatmap to identify False Positives (predicting fast when the subject was slow).
- **Feature Importances Plot**: Specifically extracted from the Random Forest's Gini impurity metrics to visually inform the user *which* physical traits (e.g. Peak Ankle Velocity vs. Peak Wrist Acceleration) were mathematically most important in determining a fast jog.

---
*Note: This architecture successfully proved the viability of multimodal sensor fusion, leading directly to the expansion of the code in Phase 2 for unstructured, high-velocity real-world athletic sprinting.*
