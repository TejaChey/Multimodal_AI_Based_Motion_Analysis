# ICICF-2027 Paper 768: Claims Audit and Revision Plan

Audit date: 2026-11-10. I checked the draft PDF ("Multimodal AI-Based Motion Analysis for Sprint Running Performance Classification Using Smartphone Technology") against the code, data, saved results and MLflow runs in this repo.

> **Title mismatch:** The acceptance email is for a different paper, *"Production-Ready MLOps Pipeline for Smartphone-Based Sprint Biomechanics: From Research Prototype to Scalable Coaching Platform"*. Reviewer comments 6–9 cover the 95 ms mobile latency, KS-test drift detection, automated retraining and 99.5% availability. None of those claims appear in this PDF, and none appear anywhere in the repo. We need the version that was actually submitted.

---

## 1. Summary

| Area | Paper says | Repo shows |
|---|---|---|
| Sprint dataset | 9 athletes, 18–27 samples | **9 recordings from 8 athletes**: Yeshwanth appears twice. All were recorded on 2026-03-25 within about 1 hour. |
| Skill labels | Assigned by coaches from 100 m times | **K-Means clusters** computed on the same 23 features, then named by `stride_freq_hz` (`train_sprint_model.py`). This is circular, and Yeshwanth's two trials got different labels (Intermediate vs Elite). |
| Headline accuracy | MLP 82%, P/R/F1 0.81 | MLflow: Rotation A **55.6%** (9 real test samples), B **100%** (3 test samples, leaked), C **66.7%** (3 samples). No run produces 82%. |
| Modality ablation (sprint) | Video 70 / IMU 55 / Fused 82 | LOOCV RF: Video **55.6** / IMU **77.8** / Fused **66.7**. IMU-only scores best, and fusion is *worse* than IMU alone. |
| Modality ablation (UTD-MHAD) | 75 / 50 / **87** | 75 / 50 / **62.5** (`modality_comparison_3class.csv`). Fusion is worse than video alone. |
| UTD-MHAD labels | Skill levels | Tercile split of `hip_angle_mean`, which is itself a video feature. This is circular, which explains why video-only wins. |
| Train/test protocol | 70/15/15 subject-wise split | RF uses LOOCV on recordings. MLP holds out 1 real sample per class. Synthetic samples are generated from **all 9** real samples *before* splitting, so every test sample's neighbours are in training (leakage). |
| Deployed model | — | The app ships **Rotation A** (55.6%). |
| Synchronisation | Audio clap marker | The code aligns the IMU window with the highest accel magnitude to the video duration (`sync_trim_imu.py`). No audio clap is used. |
| Venue | Indoor track | Footage is outdoors: open sky, trees, stands. |

---

## 2. Claim-by-claim status

Legend: ✅ supported · ⚠️ partly/needs evidence · ❌ contradicted by repo · ❓ team must confirm

| # | Claim in draft | Status | Evidence / note |
|---|---|---|---|
| 1 | IMU sampled at 100 Hz | ✅ | Median 99.3–99.6 Hz across all 18 CSVs |
| 2 | Video at 60 fps | ✅ | ffprobe: 60/1, 1920×1080 HEVC, with an AAC audio track |
| 3 | iPhone 13, tripod, 10 m away, 1 m high | ❓ | The file metadata has no device model. Confirm with whoever filmed. |
| 4 | Galaxy Watch 4 on non-dominant wrist | ❓ | CSV format matches the Sensor Logger app. Confirm the watch model and wrist. |
| 5 | 9 athletes | ❌ | 8 unique people, 9 recordings |
| 6 | 18–27 sprint samples | ❌ | 9 |
| 7 | Ages 18–30, track teams/clubs | ❓ | No record in the repo |
| 8 | Ethics approval and signed consent | ❓ | **Must be true before publication.** If you have no approval, say "informed verbal consent; institutional approval pending/exempt" only if that is accurate. |
| 9 | Coach labels from 100 m times | ❌ | K-Means (see above) |
| 10 | Audio-clap synchronisation | ❌ | Intensity-window alignment |
| 11 | 35 m track, 15 m fly zone | ⚠️ | Clips last 2.3–3.6 s. The athlete crosses the camera's field of view, and no cones are visible. |
| 12 | Stride length via reference-object calibration | ⚠️ | Values cluster at 2.5 / 3.0 m, which suggests heavy quantisation. Check `process_sprint_videos.py`. |
| 13 | 23 features | ✅ | Includes `duration_s` and `total_frames`, which are clip-length artefacts, not biomechanics |
| 14 | RF 100 trees, max_depth 5 | ⚠️ | Code has no max_depth |
| 15 | SVM RBF C=1 | ⚠️ | No SVM run on sprint data in the repo |
| 16 | SMOTE k=5 | ⚠️ | Impossible with 2–4 samples per class; the code lowers k |
| 17 | MLP 128-64-32, dropout, BN, L2, Adam 1e-3 | ✅ | Matches the code |
| 18 | 100 epochs, patience 10 | ❌ | Code uses 200 epochs and patience 20 |
| 19 | UTD-MHAD multimodal 87% | ❌ | 62.5% |
| 20 | RF base 66% | ✅ | 66.7% LOOCV, on K-Means labels |
| 21 | RF aug 72%, SVM 58%/68%, MLP 82%, P/F1 | ❌ | No source anywhere in the repo |
| 22 | Fig. 6 confusion matrix | ❌ | Shows 100% on all 9 real samples, which contradicts the 82% claim. It is Rotation A-style testing on samples whose synthetic copies were in training. |
| 23 | Fig. 3 "96.67% accuracy" | ❌ | No source. It also names a K-Means autograder, which contradicts the coach-label text. |
| 24 | 47 ms inference on i7, 2.5 s video processing | ⚠️ | Never measured in code. Easy to measure for real. |
| 25 | ±5° joint-angle accuracy | ⚠️ | Not validated here; only cited from other work. Phrase it as "reported in [x]". |
| 26 | Under $500, ~200× cheaper | ✅ | A cost estimate; fine as long as it's framed as one |
| 27 | Feature importance: GCT and peak accel on top | ⚠️ | Re-derive after the labels are fixed |

### Text problems the reviewers didn't flag but will hurt in IEEE review
- **Citations don't match the reference list.** Weyand is cited as [15], but [15] is Cust et al. VideoRun2D is [11], but [11] is Kim et al. "Yang and Park" is [12], but [12] is Tan et al. Kidzinski is [16], Slawinski [6], Taborri [8], and so on. Every reference needs to be checked to confirm it exists and matches the claim. Some look possibly fabricated.
- Discussion §V-B contains a garbled paragraph: *"adaptive-waveform ESM-COTS operations … CoN … CMA project"*. Delete it.
- §II-D "Gap Analysis" is a heading with no content.
- Fig. 2 has a stray "Tap to enter" UI overlay.
- The abstract says "accuracy rate of 87" (no %), "85%" vs 82% in the body, and "12.5% better" vs +12% in Table III. These need to be made consistent.
- "Olympic-level performance tracking" is an overclaim for n = 8.
- The jerk paragraph's "opposite of what … manufacturers claim" makes no sense and needs rewording or removal.

---

## 3. Reviewer comments: what we can make true vs. what we must rewrite

The rule: **a claim becomes true by running the experiment and reporting what comes out.** It never becomes true by tuning until the number hits 82%. Some claims (coach labels, audio clap, 18–27 samples, 99.5% uptime) describe things that didn't happen. Those get rewritten to describe what did happen.

| # | Comment | Honest fix (within ~7 days) | Effort |
|---|---|---|---|
| 1 | Only 9 athletes | **Best:** record more athletes with the same protocol (one session at the DSU track, aiming for 20–30). **Minimum:** reframe as a pilot/feasibility study (n = 8 athletes, 9 trials), add per-class counts and a power/limitations paragraph, and move generalisation claims to future work. | Days (data) / hours (text) |
| 2 | Subject-independent protocol unclear | Re-run with **Leave-One-Subject-Out** (Yeshwanth's trials grouped together). Do **all scaling and augmentation inside each training fold** so test athletes never seed synthetic data. Add a protocol figure. | ~1 day of code |
| 3 | Annotation/labelling criteria | Replace K-Means with an **objective label**. **(a)** Fly-segment velocity measured from video: frames to cross two fixed pixel lines, scaled by a known track distance. This is valid if the camera was in the same spot for every athlete. **(b)** Real 100 m PBs, if you have them. Then remove `duration_s`/`total_frames` from the features, or the label leaks. Document the thresholds. | ~1 day |
| 4 | Validate 82% with P/R/F1 and confusion matrix | Report what the LOSO re-run gives: accuracy, macro P/R/F1, a pooled confusion matrix, bootstrap 95% CIs, and a **permutation-test p-value**. With n ≈ 9, the number will move. Report whatever it is, and replace 82% in the abstract and conclusion. | Falls out of #2 |
| 5 | Video + IMU integration | Describe the real sync method. Add a validation: cross-correlate the MediaPipe **wrist landmark** acceleration with the **watch** accelerometer (same body part) to estimate the time offset and residual error per athlete. Describe the fusion as feature-level (early) concatenation, since that's what the code does, not "late fusion". | ~1 day |
| 6 | 95 ms mobile latency across devices | Export the MLP to TFLite and time MediaPipe Pose + feature extraction + MLP on **2–3 real phones** plus a laptop. Report mean/p95 per stage and device. If you can't, state the measured laptop numbers and drop "mobile". | ~1 day if phones are available |
| 7 | KS-test drift thresholds | Nothing exists yet. Either implement it (per-feature 2-sample KS with Bonferroni-corrected α = 0.05/23, a minimum window size justified by KS power, PSI as a secondary signal) or remove the claim. | ~1 day |
| 8 | Retraining under changing populations | Implementable only as a **simulation**: train on a subset of athletes, stream held-out athletes plus synthetic shifts, and show drift detection and post-retrain accuracy. It must be labelled as simulated. With 8 people it is weak, so softening to "framework designed; evaluation is future work" is safer. | 1–2 days, or a text change |
| 9 | 99.5% availability and scalability | There's no evidence, and a 7-day window can't produce a meaningful 99.5% figure. **Remove**, or replace with a measured load test (e.g. Locust against the Streamlit app: requests/s and p95 latency at N users) and state the hosting platform's SLA as the platform's, not ours. | Hours |
| 10 | Reproducibility | Add a table covering devices (phone model, watch model, Sensor Logger version), Python 3.11, mediapipe 0.10.9, TensorFlow ≥2.16.1, scikit-learn 1.7.2, seeds (42), `params.yaml` hyperparameters, and training hardware. Pin all versions in `requirements.txt` and publish the code (and data, if consent allows). | Hours |

---

## 4. Questions the team needs to answer
1. Can you send the **submitted** MLOps-titled manuscript? Comments 6–9 refer to it.
2. Do you have real **100 m times or coach ratings** for the athletes? If not, was the camera in the **same fixed position** for all clips? That decides whether velocity-based labels work.
3. Is there **ethics approval / signed consent**? This determines what the paper may say and whether data can be released.
4. Can you **record more athletes** this week?
5. Which phones can you borrow for the latency benchmark?
