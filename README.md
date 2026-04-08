# Multimodal AI-Based Motion Analysis 🏃‍♂️⌚

A comprehensive machine learning pipeline that fuses high-speed optical computer vision (MediaPipe) with wearable telemetry (Samsung Galaxy Watch IMU) to mathematically classify and grade human athletic biomechanics.

This project was developed in two distinct phases, scaling from controlled academic datasets to unstructured, high-velocity real-world athletic sprinting.

---

##  Project Overview

### Phase 1: Academic Baseline (UTD-MHAD)
**Objective**: Establish a baseline multimodal sensor fusion architecture.
* Utilized the [UTD-MHAD Dataset](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) to classify "Jogging in Place" action intensities (Fast vs. Slow).
* Fused 30fps MediaPipe skeletal joint velocities with wearable 3D acceleration thresholds.
* Successfully proved that Late Fusion machine learning (Random Forest & SVM) could reliably classify human motion using multidimensional data streams.

### Phase 2: High-Velocity Real-World Sprinting
**Objective**: Build a 3-tier athletic Autograder (Elite, Intermediate, Beginner) evaluating custom real-world 15-meter fly zone sprints.
* **The Dataset**: Custom dual-modality data collected from 9 athletes. Captured via 60fps smartphone cameras and high-G Samsung Galaxy Watch telemetry (dynamic 50-100Hz).
* **Clock-Drift Synchronization**: Bypassed unreliable UNIX timestamps by developing a **Savitzky-Golay Intensity Overlap Algorithm**, which structurally mapped continuous peak exertion IMU forces directly onto the video duration.
* **Algorithmic Proxies**: Engineered 23 biomechanical features without expensive hardware. This includes calculating **Ground Contact Time (GCT)** through vertical ankle velocity, extracting **Stride Cadence (Hz)** using Welch's Power Spectral Density on the gyroscope, and grading biomechanical efficiency using **SPARC** (Spectral Arc Length) motion smoothness.
* **The ML Engine**: Grouped athletes using Unsupervised **K-Means Clustering** to remove human bias, and validated the classifier using **Leave-One-Out Cross-Validation (LOOCV)** to ensure scientific validity on small sample sizes.

### Phase 3: Physics-Bound Data Augmentation & Deep Learning
**Objective**: Overcome the "Curse of Dimensionality" caused by small datasets ($N=9$) using physics-based synthetic data generation.
* **Data Augmentation**: Developed a 4-method pipeline (Gaussian Variance, Linear Interpolation, Biomechanical Jitter, and SMOTE) to synthetically expand the dataset to **900 athletes** while strictly enforcing physiological boundaries and logic (e.g. Flight Ratio + GCT Ratio strictly equaling 1.0).
* **Deep Learning Ablation Study**: Trained a Multi-Layer Perceptron (MLP) neural network. Conducted a rigorous data rotation ablation study which mathematically proved that a **Mixed Data Rotation** (900 synthetic samples anchored by 6 real original samples) produced a flawless **100% test accuracy** on unseen original athletes.

### Phase 4: Full-Stack Web Application (Deployment)
**Objective**: Deploy the `best_sprint_model.keras` weights into a real-time interactive user interface.
* **Streamlit Dashboard**: A highly customized, Glassmorphism-styled dark-mode Python Web App.
* **Inference Pipeline**: Users drag-and-drop a video and IMU file. The app asynchronously pipes the data through the Phase 1 and 2 extractors, Normalizes the inputs using `joblib`, and executes the Deep Learning neural pass in real time.
* **Expert Coaching Engine**: Custom Python logic compares the user's specific biomechanical outputs against the "Elite" dataset mathematical averages and dynamically generates localized coaching tips (e.g. noting if a user's *Stride Frequency* deviates beyond baseline norms).

---

## 📂 Project Structure

```
Multimodal-AI-Based-Motion-Analysis-/
├── Phase_1/                          # Academic Baseline Pipeline
│   ├── code/                         # UTD-MHAD feature extraction and baseline ML models
│   ├── data/                         # UTD-MHAD dataset storage
│   └── results/                      # Phase 1 models and confusion matrices
│
├── Phase_2/                          # Real-World Sprint Analysis Engine
│   ├── code/
│   │   ├── sync_trim_imu.py          # Savitzky-Golay clock synchronization
│   │   ├── process_sprint_videos.py  # 60fps MediaPipe Video kinematics
│   │   ├── process_sprint_imu.py     # High-G Wearable telemetrics & Spectral Analysis
│   │   ├── merge_sprint_features.py  # 23-dimension Late Fusion concatenation
│   │   ├── train_sprint_model.py     # K-Means Autograder & LOOCV Random Forest
│   │   └── generate_extra_visuals.py # Modality comparison & ROC AUC curve generation
│   ├── data/
│   │   └── sprint_raw/               # Unstructured .mp4 and Galaxy Watch .csv files
│   └── results/
│       ├── synced_imu/               # Output sync-overlap graphs
│       ├── video_features/           # Skeletal tracking data
│       └── models/                   # Final models, ROC curves, and Feature Importances
│
├── Phase_3/                          # Deep Learning & Augmentation
│   ├── code/
│   │   ├── data_augmentation.py      # Generates 900 physics-bound synthetic samples
│   │   └── train_dl_model.py         # Multi-Layer Perceptron (MLP) & Ablation Study
│
├── App/                              # Real-Time Web Deployment
│   ├── app.py                        # Streamlit UI Dashboard
│   ├── coaching_engine.py            # Rule-based Expert System for feedback
│   └── inference.py                  # API Bridge for Keras Model inference
│
├── requirements.txt                  
└── README.md
```

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone <repo-url>
cd Multimodal-AI-Based-Motion-Analysis-

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃🏽‍♀️ Running the Phase 2 Sprint Pipeline

Execute the pipeline in sequential order with the virtual environment activated:

```bash
# 1. Mathematically synchronize the Watch IMU hardware to the Video
python Phase_2\code\sync_trim_imu.py

# 2. Track 60fps 3D Kinematics and algorithmic GCT
python Phase_2\code\process_sprint_videos.py

# 3. Extract Spectral (PSD) Stride Frequencies from Watch forces
python Phase_2\code\process_sprint_imu.py

# 4. Fuse both datasets via Late Fusion
python Phase_2\code\merge_sprint_features.py

# 5. Run the K-Means Autograder and LOOCV Machine Learning engine
python Phase_2\code\train_sprint_model.py

# 6. Generate academic presentation graphics (ROC, Comparisons)
python Phase_2\code\generate_extra_visuals.py
```

---

## 📊 Results & Scientific Findings
* **Baseline Limitations**: The Phase 2 engine using traditional ML capped at **66.67% test accuracy** due to severe dataset starvation ($N=9$).
* **The Solution**: The Phase 3 Deep Learning **Ablation Study** mathematically proved that generating massive arrays of synthetic data ($N=900$) and anchoring it with empirical samples (**Rotation B: Mixed Data**) allowed the Neural Network to generalize perfectly, resolving the Domain Gap to hit **100% test accuracy**.

---

## 🌐 Launching the Web App (Deployment)

To launch the real-time Streamlit Artificial Intelligence dashboard locally:

```bash
streamlit run App/app.py
```
This will open up `localhost:8501` in your browser. Drag and drop any raw sprint `.mp4` and synchronized `.csv` IMU file to watch the pipeline execute in real time.

---

