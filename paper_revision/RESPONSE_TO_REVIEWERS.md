# Response to Reviewers: ICICF-2027, Paper ID 768

**Paper:** Production-Ready MLOps Pipeline for Smartphone-Based Sprint Biomechanics: From Research Prototype to Scalable Coaching Platform

> Items marked **[CONFIRM]** must be checked by the authors before sending. Delete this note and the markers afterwards.

Dear Technical Program Committee and Reviewers,

Thank you for the careful and constructive review. While addressing the comments, we re-ran every experiment from the raw recordings with a fully reproducible, subject-independent protocol. This exposed **errors in our original analysis that inflated the reported results**, and we have corrected them in the camera-ready version. In summary:

1. **Labels.** The original skill labels were produced by K-Means clustering on the same features later used for classification (circular). They have been replaced by an **objective label: each athlete's measured maximal running velocity** from the video.
2. **Leakage.** Synthetic samples had been generated from all recordings before the train/test split, so the "82%" and "100%" figures do not reflect performance on unseen athletes. Under leakage-free Leave-One-Subject-Out validation, **no classifier exceeds chance with statistical significance** (best 55.6% vs. 33.3% chance, permutation p = 0.16). We now report this honestly. The paper is repositioned as a validated measurement and MLOps pipeline plus a pilot study, rather than a solved classification problem.
3. **Data integrity.** Our new synchronisation checks found that 2 of the 9 smartwatch files do not belong to the filmed sprint, and that MediaPipe had locked onto a static false detection in early frames. Both are corrected and reported.

4. **MLOps system.** Several components the original text described (automated drift detection, retraining, a registry with a promotion gate, load and fault testing) did not exist in the code. We built and tested them (Comments 7-10). Two claims cannot be supported and are withdrawn: 99.5% availability, and any accuracy gain from retraining.

We believe these corrections make the paper more useful to the community, and we hope the committee agrees they are within the scope of minor revisions. The point-by-point responses follow.

---

### Comment 1: Only nine athletes limits generalisability.

**Response.** We agree, and we also correct the count. The dataset contains **9 sprint recordings from 8 distinct athletes** (one athlete performed two trials), all captured in one session. Collecting new data was not possible within the revision period. We have therefore:
- stated the true sample size (8 athletes, 9 trials, 3 recordings per tier) in the abstract, Section III-B and Table I;
- reframed the classification study as a **pilot feasibility study** and removed generalisation claims ("Olympic-level", "athletes worldwide");
- **quantified the data needed** for the MLOps components. With 9 reference samples, the KS drift monitor detects a 1-SD feature shift in at most 29% of cases. With 100 reference samples and a 30-sample window it detects 84% at a 5% false-alarm rate (new Fig. X, Section V-D). This gives a concrete target for future data collection;
- added a Limitations paragraph that makes clear the classification results are not evidence of generalisation.

### Comment 2: Subject-independent training and testing are not clearly explained.

**Response.** The original protocol was not subject-independent: augmented samples were generated from all athletes before splitting, so test athletes' data leaked into training. The revised protocol (Section III-F, new Fig. Y) is:
- **Leave-One-Subject-Out (LOSO)** cross-validation with 8 folds. Both trials of the athlete with two recordings are always held out together.
- Inside each fold, **using training athletes only**: standardisation is fitted, then augmentation (Gaussian jitter at 5% of feature SD plus same-class interpolation, to 300 samples per class) is applied. The held-out athlete never contributes to scaling, augmentation or early stopping.
- All 9 out-of-fold predictions are pooled for evaluation. The MLP is run with 5 random seeds; we report the mean and SD.

### Comment 3: The annotation process and expert labelling criteria require clarification.

**Response.** We must correct the original text. Labels were **not** assigned by coaches from 100 m times. They came from K-Means clustering on the classifier's own input features, which is circular. The camera-ready version uses an **objective, automatically measured label** instead (Section III-C, new Fig. Z):
- The athlete's horizontal hip velocity is measured while crossing the static camera's field of view (Theil–Sen fit, R² ≥ 0.998 for every recording).
- Velocity is normalised by a posture-invariant body length (trunk + thigh + shank ≈ 0.779 × stature), which makes it independent of camera distance and zoom.
- Measured velocities are 6.7–9.4 m/s (assuming 1.70 m stature). Tiers are cohort tertiles: Beginner < 7.45 m/s ≤ Intermediate < 7.79 m/s ≤ Elite.
- Both trials of the repeated athlete fall in the same tier, which is a consistency check.
- Features that are algebraically linked to the label (stride length, clip duration, frame count) were removed from the inputs to prevent leakage.

### Comment 4: Validate the 82% accuracy with precision, recall, F1 and a confusion matrix.

**Response.** We performed this validation, and **the 82% figure did not survive it**. Under the leakage-free LOSO protocol, with velocity labels, Table II now reports accuracy, balanced accuracy, macro precision, recall and F1, subject-level bootstrap 95% CIs, and label-permutation p-values (200 permutations). Pooled confusion matrices are shown in Fig. W.

| Modality | Best model | Accuracy | Macro P | Macro R | Macro F1 | 95% CI (acc) | Perm. p |
|---|---|---|---|---|---|---|---|
| Video-only | MLP (no aug.) | 35.6% ± 16.5 | 0.33 | 0.36 | 0.31 | 13–75% | – |
| IMU-only | LogReg + aug. | 55.6% | 0.42 | 0.56 | 0.48 | 25–88% | 0.16 |
| Fused | MLP (no aug.) | 28.9% ± 16.9 | 0.23 | 0.29 | 0.24 | 13–89% | – |

Chance is 33.3%. No configuration is significantly above chance, and fusion does not outperform the single modalities. A LOSO ridge regression of velocity also fails to beat a mean predictor. With 8 athletes, skill-tier classification is not yet reliable. The earlier 82%/100% figures were artefacts of the leakage described under Comment 2, and we state this explicitly in Section IV.

### Comment 5: The integration of smartphone video and Galaxy Watch IMU requires further technical explanation.

**Response.** Section III-D has been rewritten and a quantitative validation added (new Fig. V):
- **Acquisition.** Video: Samsung Android smartphone **[CONFIRM model]**, 1920×1080 HEVC at 60 fps, tripod, side view. IMU: Samsung Galaxy Watch **[CONFIRM model and wrist]**, triaxial accelerometer and gyroscope at 99.3–99.6 Hz (measured), recorded with the Sensor Logger app. *(The original text's "iPhone 13" and "audio clap" synchronisation were incorrect; the video metadata identifies an Android device and no clap marker was used.)*
- **Temporal alignment.** The IMU segment is the highest-intensity window whose length equals the time the runner is tracked in the video. Accelerometer and gyroscope streams are merged by nearest timestamp (tolerance 20 ms).
- **Validation 1, clock consistency.** For 7 of 9 recordings, the video creation time follows the watch's last sample by a consistent 1.6–5.7 s (median 2.1 s). The other two differ by −102 s and −422 s, showing that those watch files were recorded during a different run. They are reported, a sensitivity analysis excludes them, and the same checks now run **automatically** as a data-quality gate in the pipeline and in the API, which quarantined both files without manual inspection.
- **Validation 2, cadence agreement.** Stride frequency from the video wrist landmark agrees with the watch gyroscope: MAE 0.073 Hz, bias −0.05 Hz, r = 0.95 (n = 7).
- **Validation 3, waveform agreement.** The cross-correlation between video wrist velocity and the watch's dominant gyroscope axis has |r| of 0.51–0.94 (median 0.76).
- **Fusion.** Fusion is early (feature-level) concatenation of 9 video and 10 IMU features after per-fold standardisation. We corrected the original "late fusion" description.

### Comment 6: The 95 ms mobile inference latency needs validation across devices.

**Response.** We replaced the unmeasured figure with **measured per-stage latency** (new Table III) on two hardware configurations, using a released benchmark script so others can reproduce it on their own devices. Values are medians, with p95 in parentheses.

| Stage | Apple M5, all cores | Apple M5, efficiency cores only |
|---|---|---|
| Video decode (per frame) | 1.4 ms | 10.6 ms |
| MediaPipe Pose, lite (per frame) | 7.1 (13.0) ms | 45.8 (75.8) ms |
| MediaPipe Pose, full: deployed (per frame) | 11.5 (17.9) ms | 61.3 (114.2) ms |
| MediaPipe Pose, heavy (per frame) | 49.3 (56.3) ms | 212.3 (273.3) ms |
| Feature extraction (video + IMU, per sprint) | 8.1 ms | 49.9 ms |
| Classifier, Keras `predict()` | 14.5 ms | 95.3 ms |
| Classifier, TFLite, 1 thread (52 KB model) | 0.001 ms | 0.009 ms |
| End-to-end per sprint clip (full model) | 2.1 s | 12.7 s |

Pose estimation dominates the cost; the classifier itself is negligible once exported to TFLite. The 95 ms figure matches Keras framework overhead on a low-power core, not model compute. Through the Docker API, a complete request (upload, pose, features, classifier) has a median latency of 2.1 s per clip on one user. We removed the unqualified "mobile" claim: on-phone execution was not measured, and we now present it as future work. **[If a team member runs `06_latency.py` on another laptop or phone, add a column.]**

### Comment 7: The KS-test drift-detection thresholds and retraining triggers lack justification.

**Response.** The drift monitor and retraining triggers are now implemented in the running system (`sprintml.monitor`, a monitor/retrain worker, and the `/drift` endpoint), and each threshold is justified:
- **Per-feature test.** Two-sample KS against the training reference, Bonferroni-corrected, α = 0.05/19 = 0.0026, which bounds the family-wise false-alarm rate at 5%. A unit test confirms ≤ 10% false alarms over 300 null windows (nominal 5%). PSI is reported as a secondary signal (alert at 0.25).
- **Window and reference size.** A Monte-Carlo power analysis (Fig. X) shows that a 1-SD shift is detected with 84% probability only with a reference ≥ 100 and a window ≥ 30. **Alarms are therefore armed only above those sizes**; below them the monitor still reports p-values but labels them *informational*. Our current reference has 7 recordings, so no alarm is armed today, and we state this openly.
- **Retraining triggers.** (i) an armed drift alarm; (ii) every 3 newly collected, quality-passed recordings; (iii) manual. The value 3 is a pragmatic setting (a parameter in `params.yaml`), safe because every retrain passes a promotion gate (comment 8): a poor challenger is never served, so frequent retraining carries little risk.
- **Replay on real recordings through the API** (7 quality-passed clips):

| Shift | What happens |
|---|---|
| none | no alarm |
| watch at 50 Hz | all 7 rejected by the data-quality gate (never reach the monitor) |
| accelerometer gain ×1.5 | informational alarm on `peak_accel`, `std_accel`, `rms_jerk` (p = 0.00058 each); not armed |

### Comment 8: The effectiveness of retraining under changing athlete populations needs validation.

**Response.** We implemented automated retraining (feature store → trigger → leave-one-subject-out evaluation → **promotion gate** → registry → hot reload) and replayed the 9 recordings in real capture order from a cold start. Labels for new recordings come automatically from measured velocity.
- Recordings that fail the quality gate (2 of 9) are quarantined and never trained on.
- A model first becomes trainable at the 8th arrival (≥ 2 recordings per tier from ≥ 5 athletes). It is registered as **provisional** (LOSO accuracy 66.7%, permutation p = 0.099, not significant).
- At the 9th arrival the retrained challenger is **refused** (p = 0.65, macro-F1 0.24 < 0.56), so the served model is protected.

We cannot claim that retraining *improves accuracy*: only one arrival could be scored by a champion before training. What we demonstrate is that the automation works and that the gate prevents unvalidated models from replacing the served one. Evaluating accuracy gains under population change requires the larger dataset described in Section VI.

### Comment 9: The 99.5% availability and scalability claims need supporting experimental evidence.

**Response.** We removed the 99.5% claim, which had no evidence, and replaced it with measurements on the Docker deployment (Table V):
- **Throughput/latency** (Locust, real 2–3 s clips including upload, 214 requests, 0 failures): 0.48 req/s at 2.1 s median with one user; saturating at ≈ 1.1–1.2 clips/s (≈ 70/min) at 3.3–6.5 s median under 4–8 concurrent users on a 10-core node. **Adding workers did not increase throughput** (1.21, 1.14, 1.04 req/s for 1, 2, 4 workers) because MediaPipe already uses several cores per request; scaling further requires more nodes, which we did not test.
- **Fault tolerance:** killing 1 of 2 workers under continuous probing caused 0 failed probes and a replacement in 0.03 s; a crash of the server process was recovered by the container restart policy in 0.9–1.0 s; a container restart took 3.8–3.9 s.
- **Availability:** 60/60 health probes succeeded over a 5-minute fault-free window (100%, Wilson 95% lower bound 94%). This window is far too short to support a figure near 99.5%, so we make no such claim. **[Optional: run `scripts/uptime_probe.py` until submission and quote the measured window.]**

### Comment 10: Hardware, software versions and experimental settings should be documented.

**Response.** Added as Table IV, plus a reproducible, versioned pipeline:
- **Data and code lineage:** a DVC pipeline (`dvc repro`) with the raw-data hash recorded in `dvc.lock`; every MLflow run logs the git commit, data hash, hyper-parameters, metrics and the evaluation report.
- **Software:** Python 3.11.17, MediaPipe 0.10.9 (Pose, complexity 1), scikit-learn 1.7.2, NumPy 1.26.4, pandas 2.3.3, SciPy 1.17.1, OpenCV 4.11, MLflow 3.5.1, DVC 3.67.1, FastAPI 0.162.2, TensorFlow 2.18.1 (latency benchmark only). Exact versions: `paper_revision/requirements-lock.txt`; serving image: `requirements-serve.txt` and `Dockerfile`.
- **Hardware:** Apple M5 (10 cores), 16 GB RAM, macOS 27.0.1; Docker Desktop VM (10 CPUs, 8 GB).
- **Settings:** all parameters in `params.yaml` (seed 42; MLP seeds 42–46); 43 automated tests run in about 9 s and in CI.
- **Platform note:** pose numbers differ slightly between macOS and the Linux container (e.g. 8.30 vs 8.32 m/s), so we state the platform with each result.

**[CONFIRM: repository URL; whether the raw data can be shared under the participants' consent.]**

---

**Additional corrections made at our own initiative.**
- Fixed in-text citation numbering, which did not match the reference list, and verified every reference. **[TODO: not done yet. Do this before sending.]**
- Removed an erroneous paragraph in Section V-B and the empty Section II-D.
- Corrected the venue description from indoor to outdoor track.
- Removed UTD-MHAD "skill" classification results, whose labels were derived from an input feature.
- Fixed the static-detection tracking error in the pose pipeline.

**[CONFIRM ethics statement wording: approval number, or the exact consent procedure followed.]**

Sincerely,
The Authors
