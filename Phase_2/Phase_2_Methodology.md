# Phase 2: Multimodal Sprint Biomechanics Methodology

## 1. Project Objective
Following the success of Phase 1's preliminary jogging model, Phase 2 scaled the architecture into unpredictable, high-velocity real-world physics. The objective was to capture and classify maximal-effort **15-meter fly zone sprints** to automatically grade human athletic performance into three tiers: **Elite, Intermediate, and Beginner**.

## 2. Dataset Overview
Instead of a controlled academic dataset, Phase 2 utilized custom-collected real-world athletic data across 9 subjects.

**Data Streams Utilized:**
1. **High-Speed Vision**: Smartphone cameras capturing 60fps lateral tracking footage.
2. **Consumer Wearable IMU**: Samsung Galaxy Watch application capturing high-G tri-axial Accelerometer and Gyroscope data at dynamic sampling rates (approx. ~50-100Hz).

---

## 3. Data Processing Architecture

### A. Advanced Synchronization (`sync_trim_imu.py`)
Consumer devices operating on different system clocks (Smartphone iOS/Android vs. Galaxy WearOS) suffer from inherent clock drift, making UNIX timestamp merging physically inaccurate. 
- **Intensity Overlap Algorithm**: We bypassed clock metadata entirely. We applied a Savitzky-Golay filter to smooth the IMU acceleration magnitude and isolated the continuous block of peak exertion (the sprint window).
- **Time Window Trimming**: We structurally overlaid this peak exertion window precisely onto the duration of the 60fps video, perfectly trimming and time-aligning both modalities without relying on system clocks.

### B. High-Speed Vision Engine (`process_sprint_videos.py`)
MediaPipe Pose was re-architected to handle 60fps high-velocity motion blur.
1. **Virtual Pressure Plates (GCT)**: Instead of expensive physical track pressure plates, we introduced a computer-vision proxy. By tracking when the vertical pixel velocity of the athlete's lowest ankle dropped to exactly zero, we mathematically calculated **Ground Contact Time (GCT)** and **Flight Time**.
2. **Kinematic Feature Extraction**: 12 specific sprint features were extracted, including:
   - Stride Length Proxies (Max horizontal foot spread)
   - Vertical Center of Mass Oscillation
   - Maximum knee drive velocity

### C. Inertial Sprint Engine (`process_sprint_imu.py`)
The Galaxy Watch captured extreme physical forces (approaching ~13.5G impacts).
1. **Spectral Frequency Analysis**: Sprinting relies heavily on cadence. We applied **Welch's Power Spectral Density (PSD)** on the raw gyroscope signal to isolate the peak frequency domain. This allowed us to mathematically extract the athlete's exact **Stride Rate (Hz)** without relying on optical foot-strikes.
2. **Smoothness Grading**: We ran **SPARC (Spectral Arc Length)** formulas to grade the biological "smoothness" and mechanical efficiency of the arm drive.
3. **Total Feature Count**: 11 IMU features were aggregated per athlete.

---

## 4. Machine Learning & Classification (`train_sprint_model.py`)

A massive challenge in Phase 2 was the limited sample size (9 athletes). We could not use a traditional 80/20 Train/Test split, as a Test set of 2 athletes would be statistically invalid. 

1. **Feature Fusion**: The 12 Video and 11 IMU features were mathematically merged into a 23-feature dataset and scaled using `StandardScaler`.
2. **The K-Means Autograder**: Without explicit prior labels for who was "Elite" vs "Beginner", we deployed an unsupervised **K-Means Clustering** engine (k=3). It analyzed the 23-feature space and objectively grouped the athletes based on their biomechanical superiority (anchored heavily by their raw internal stride frequency potential).
3. **Leave-One-Out Validation (LOOCV)**: To scientifically evaluate the model on only 9 samples, we used Leave-One-Out Cross-Validation. The **Random Forest Classifier** trained on 8 athletes and tested on the 1 remaining athlete, looping 9 times.
4. **Final Accuracy**: The model achieved an impressive **66.67%** LOOCV accuracy, successfully predicting the exact skill tier of a physically unseen athlete at double the rate of random probability (33%).

## 5. Output Artifacts
- **Confusion Matrix**: Highlighting the boundary prediction overlap between Beginners and Intermediates.
- **Top 10 Feature Importances**: Demonstrating that fused inputs (e.g. Video Stride Frequency + IMU Peak Accel) were physically responsible for dictating athletic speed.
