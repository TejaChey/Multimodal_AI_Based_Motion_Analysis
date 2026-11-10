# MLOps Implementation Plan (using only the existing 9 recordings)

Goal: build the MLOps system the accepted paper describes, so that every MLOps claim in the camera-ready describes something that **exists, runs and was measured**.

**What the existing data can and can't prove.** It can prove that each component works mechanically: drift is detected, retraining triggers, bad models are blocked, the service scales and recovers. It cannot prove that retraining *improves accuracy*, because 8 athletes is too few. The paper must keep those two things separate.

---

## Where we are today

| Exists | Missing |
|---|---|
| `params.yaml`, MLflow tracking (3 runs), registered `Sprint_MLP_Model` v1 | Data versioning, pipeline orchestration |
| Streamlit app, devcontainer | API service, Docker image, CI/CD, tests |
| Corrected pipeline in `paper_revision/code/` (labels, features, LOSO, drift analysis) | Drift monitoring inside the running system, automated retraining, promotion gates, uptime/load evidence |
| — | The app still serves the leaked model with the old features |

## Target architecture

```
 phone video + watch CSVs
            │
            ▼
 ┌──────────────────────┐   per-request log    ┌──────────────────────┐
 │ FastAPI service      │ ───────────────────▶ │ Feature store        │
 │ /predict /health     │                      │ (SQLite/Parquet)     │
 │ /metrics (Prometheus)│                      └──────────┬───────────┘
 └──────────▲───────────┘                                 │ scheduled job
            │ loads @champion                             ▼
 ┌──────────┴───────────┐  promote if gate passes  ┌──────────────────────┐
 │ MLflow registry      │ ◀─────────────────────── │ Monitor + retrain    │
 │ champion/challenger  │                          │ KS drift, data QA,   │
 └──────────────────────┘                          │ auto-label (velocity)│
            ▲                                      └──────────┬───────────┘
            │ register                                        │ triggers
 ┌──────────┴───────────────────────────────────────────────▼───────────┐
 │ DVC pipeline: pose → labels → features → validate → train → evaluate │
 └──────────────────────────────────────────────────────────────────────┘
 CI (GitHub Actions): tests → pipeline smoke run → build Docker → deploy
```

---

## Phases

Ordered so the work most needed for the camera-ready comes first. Effort is in working days.

### P1. Consolidate the pipeline (1 day)
- Move the corrected code from `paper_revision/code/` into `src/` as importable modules: `pose`, `labels`, `features`, `validation`, `train`, `evaluate`, `monitoring`.
- Put every constant in `params.yaml`: stance threshold, augmentation settings, KS α, window sizes.
- Retire the old 23-feature extractors in `src/data/`.
- **Done when:** one command reproduces `paper_revision/results/` exactly.

### P2. Data versioning + reproducible pipeline (0.5 day) → comment 10
- DVC tracks `Phase_2/data/sprint_raw/` (local or Google Drive remote) and defines a `dvc.yaml` with stages pose → labels → features → validate → train → evaluate.
- Every MLflow run logs the git commit, the DVC data hash, `params.yaml` and the lock file.
- **Done when:** `dvc repro` rebuilds everything, and any MLflow run can be traced to its exact data and code.

### P3. Automated data-quality gate (0.5 day) → comment 5
- The `validate` stage checks each recording automatically:
  - video ≥ 60 fps and IMU 90–110 Hz;
  - watch–video clock offset within 5 s of the cohort median;
  - video↔watch cadence agreement ≤ 0.2 Hz;
  - runner track R² ≥ 0.99;
  - no NaN features.
- A failing recording is quarantined: logged and excluded from training, never silently used.
- **Done when:** the gate flags Chandra1 and Jeeva on its own.

### P4. Training with a promotion gate (1 day) → comments 2, 4
- Training runs the LOSO evaluation and logs accuracy, macro P/R/F1, confusion matrix, CI and permutation p to MLflow.
- Use MLflow aliases `@champion` and `@challenger`. A challenger is promoted only if:
  - its macro-F1 ≥ the champion's;
  - its permutation p < 0.05;
  - it passes the data-quality gate.
- A model card (data version, metrics, limitations) is auto-generated per version.
- **Done when:** retraining on today's data produces a challenger, the gate correctly **refuses** to promote it (p = 0.16), and the reason is logged.

### P5. Serving: API + Docker + TFLite (1.5 days) → comments 6, 9
- FastAPI service:
  - `POST /predict` (video + accelerometer + gyroscope files) returns tier, features, quality flags and model version;
  - `GET /health`;
  - `GET /metrics` exports Prometheus latency histograms per stage, error counts and model version.
- The model is loaded from the MLflow `@champion` alias and hot-reloaded on promotion.
- The Streamlit app becomes a thin client of the API.
- Dockerfile + docker-compose with the API, MLflow server and Prometheus (optional), plus `restart: always` and a healthcheck.
- Export TFLite for on-device use (already built: 52 KB).
- **Done when:** `docker compose up` serves predictions end to end.

### P6. Monitoring + drift detection in production (1 day) → comment 7
- Every request's features go to the feature store.
- A scheduled monitor job runs:
  - per-feature KS test, Bonferroni α = 0.05/19;
  - PSI as a secondary signal;
  - data-quality drift checks (fps, IMU rate, offsets).
- It writes an HTML drift report and exports p-values to `/metrics`.
- Following the power analysis already done, alarms are **armed only when** reference n ≥ 100 and window m ≥ 30. Below that, the monitor reports "insufficient data" instead of false confidence.
- **Done when:** replaying the real device-shift scenarios through the API makes the monitor fire on the 50 Hz watch scenario, as it does offline.

### P7. Automated retraining loop (1 day) → comment 8
- Retraining triggers: (a) a drift alarm when armed, (b) every N new quality-passed recordings, (c) manual.
- New recordings are labelled automatically from measured velocity, with no annotator in the loop.
- Pipeline: retrain → LOSO evaluate → promotion gate → register → API reloads the champion.
- Orchestration: a GitHub Actions cron job or a small Prefect flow.
- **Done when:** the chronological replay of the 9 recordings runs end to end through the real system, with triggers, retrains and gate decisions all logged. The expected and honest result is that the loop runs but accuracy doesn't improve at this size.

### P8. CI/CD + tests (1 day)
- pytest coverage:
  - feature extraction on one fixture clip (golden values);
  - the runner-track filter;
  - each data-quality rule;
  - the KS monitor's false-alarm rate on null data;
  - the promotion-gate logic;
  - an API contract test.
- GitHub Actions on every push: lint, tests, a pipeline smoke run on 2 clips, then build and push the Docker image.

### P9. Scalability and availability evidence (1 day + a waiting period) → comment 9
- **Load test** (Locust) against the Docker API with 1/2/4 workers and 1–20 concurrent users. Report throughput (clips/min), p50/p95 latency and error rate. This is real, measured scalability evidence.
- **Recovery test:** kill the container and measure time to healthy (Docker restart policy).
- **Uptime:** an external health check every 5 minutes for the remaining days before submission. Report the **measured** uptime and window, e.g. "100% over 5 days, 1440 checks". Do **not** report 99.5% unless it is what was measured, and say the window is short.

### P10. Mobile latency (optional; needs an Android phone) → comment 6
- Run TFLite's `benchmark_model` and MediaPipe Pose on a team member's phone over `adb`, and add the result as a column in the latency table. Without a phone, the paper says on-device execution is future work.

---

## Claims after implementation

| Reviewer item | What we can truthfully write |
|---|---|
| 5 Integration | Automated data-quality gate; cadence MAE 0.07 Hz; mismatched files auto-quarantined ✅ |
| 6 Latency | Measured per stage on M5 (2 configurations) ✅, plus a phone if P10 is done |
| 7 KS thresholds | Implemented in the running service; thresholds justified by power analysis; arming rule ✅ |
| 8 Retraining | Loop implemented and exercised end to end on real chronological and device-shift streams ✅. Accuracy improvement **not demonstrated** ❌ (stated as a limitation) |
| 9 Availability/scale | Measured throughput, p95 latency and recovery time ✅. Measured uptime over a short window ⚠️. **No 99.5% claim** unless measured ❌ |
| 10 Reproducibility | DVC + MLflow lineage, Docker image, lock file, CI ✅ |
| 4 Accuracy | Unchanged: not above chance at n = 8 ❌. MLOps does not fix the dataset size. |

## Schedule

About 8–9 working days for one person. If the camera-ready is due sooner, do **P1–P6 + P9 load test** (~6 days) and describe P7 and P8 as implemented but evaluated only through the replay. Split between two people:
- **Person A:** P1–P4 (pipeline, data, training).
- **Person B:** P5, P6, P8, P9 (serving, monitoring, CI, load testing).


---

## Implementation status (2026-11-10)

P1-P9 are built and tested; **P10 (on-phone latency) was not done**. See `MLOPS_RESULTS.md` for the measured evidence.

Deviations from the plan:
- **Promotion gate:** the first model is registered as a *provisional* champion (so the service has something to serve) and flagged `validated=false`; later promotions require p < 0.05 and macro-F1 not below the champion's, exactly as planned.
- **Orchestration:** a compose `worker` service runs the monitor/retrain loop every 5 minutes instead of a GitHub Actions cron (the feature store is local to the deployment).
- **Mobile:** TFLite export from the earlier latency benchmark exists, but no phone benchmark was run.
- **Not built:** Prometheus/Grafana servers (the `/metrics` endpoint exists), GitHub Actions has not run yet (nothing pushed), the Streamlit client was edited but not launched.
