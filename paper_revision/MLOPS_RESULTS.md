# MLOps implementation: what was built and what was measured

Everything below was run on this machine (Apple M5, 10 cores, 16 GB; Docker Desktop VM with 10 CPUs / 8 GB) on **2026-11-10**, using only the 9 existing recordings. Raw outputs are in `artifacts/` (not in git); code in `src/sprintml/`, `scripts/`, `tests/`.

## 1. Status against the plan

| Phase | Built | Verified by |
|---|---|---|
| P1 package | `src/sprintml` (pose, labels, imu, features, quality, train, monitor, store, service, replay, cli) + all constants in `params.yaml` | Reproduces the validated paper features to 1e-13 (regression test on 4 clips) |
| P2 data versioning | `dvc.yaml` (pose → features → validate → train), `dvc.lock` stores the raw-data hash; every MLflow run logs git commit + data hash + params | `dvc repro` runs end to end; second run skips all stages; `dvc status` clean |
| P3 quality gate | 7 automatic checks (fps, IMU rate, runner-track R², tracked frames, finite features, video↔watch cadence, clock consistency) | Quarantines **Chandra1** (cadence 0.48 Hz off, clock −104 s) and **Jeeva** (clock −424 s); 7/9 pass |
| P4 training + gate | LOSO, scaling/augmentation inside folds, 6 candidates, max-statistic permutation p, MLflow registry with `champion`/`challenger` aliases | Gate **refuses** a non-significant challenger (p = 0.67) and keeps the champion; champion restores from the registry after `dvc repro` clears it |
| P5 serving | FastAPI (`/predict /health /metrics /drift /model /reference`), Dockerfile (1.5 GB image), `docker-compose.yml` (API + monitor/retrain worker), Streamlit app converted to a thin API client | Real clips served from the container; bad uploads → clean 422 |
| P6 monitoring | Per-feature KS (Bonferroni α = 0.05/19 = 0.0026) + PSI; alarms armed only when n_ref ≥ 100 and window ≥ 30 | Unit tests + replay through the API (section 4) |
| P7 retraining loop | Feature store (SQLite) → triggers (armed drift alarm / every 3 new passed recordings / manual) → LOSO train → gate → registry → hot-reload | Cold-start chronological replay (section 5) |
| P8 tests + CI | 43 pytest tests (~9 s); GitHub Actions workflow (tests, image build, boot smoke test) | Tests pass locally. **The workflow has not run on GitHub yet** (nothing pushed) |
| P9 evidence | Load test, fault tests, uptime probe | Sections 6–7 |
| P10 phone latency | **Not done** (no Android phone available) | – |

## 2. The deployed model, honestly

Champion v1 (registered as **provisional**): video features + logistic regression, trained on 7 recordings from 6 athletes.
LOSO accuracy 28.6 %, macro-F1 0.24, permutation p = 0.67 (200 permutations, max-statistic over 6 candidates). It is **not above chance**, so the API labels it `PROVISIONAL` and returns the **measured running velocity** (the label definition) as the primary result. Nothing in the MLOps work changes this; it only makes the system refuse to pretend otherwise.

## 3. Test suite (43 tests, 9 s)

| Area | What is tested |
|---|---|
| Runner tracking / velocity | Static false detection removed; velocity recovered to ±3 % at 3.5 / 4.5 / 5.5 statures/s; invariant to camera distance (stature 120 vs 300 px); untracked clip rejected |
| Features / IMU | 19 finite features, deterministic, loader validation |
| Quality gate | Each of the 6 rules triggers; clock check quarantines a mismatched watch file |
| Drift monitor | False-alarm rate ≤ 10 % over 300 null windows (nominal 5 %); ≥ 65 % detection of a 1-SD shift when armed; disarmed (no alarm) with a 9-sample reference; PSI; report files |
| Leakage | Held-out athlete's prediction does not depend on its own label; augmentation stays inside the class hull |
| Training / gate | Finds a planted signal (p < 0.05) and does **not** find one in noise; data-sufficiency checks; four gate cases (bootstrap, non-significant, regression, promote) |
| Registry | Promote → refuse → delete champion → restore from MLflow |
| API | 503 without a model; `/predict` contract; Prometheus counters; quality-failure 422 and logging; malformed input 422; hot reload on promotion |
| Real data | Package output equals the validated paper features (4 clips) |

## 4. Drift replay through the API (real clips, 7 quality-passed recordings)

| Scenario | Result |
|---|---|
| Baseline (unmodified) | 7/7 accepted; no alarm (smallest p = 0.99996) |
| Watch downsampled to 50 Hz | **7/7 rejected by the quality gate** (`imu_rate_hz`); the drift monitor never sees them |
| Accelerometer gain ×1.5 | 7/7 accepted; informational alarm on `peak_accel`, `std_accel`, `rms_jerk` (each p = 0.00058 < α); **not armed** (n_ref = 7 < 100) |

## 5. Retraining loop: cold-start chronological replay

The 9 recordings arrive in real capture order through the real store, trigger and gate (retrain after every new passed recording; 100 permutations).

| Arrival | Outcome |
|---|---|
| 1 Hanshi2 | passed; **no model**: a tier has no recordings |
| 2 Chandra1, 3 Jeeva | **quarantined** by the quality gate, never used |
| 4–7 Amith, Mohammed, Sachin, Tejas | passed; no model: fewer than 2 recordings in a tier |
| 8 Yeshwanth | first trainable cohort (6 recordings, 6 athletes): LOSO acc 66.7 %, p = 0.099 → registered as **provisional champion** (not validated) |
| 9 Yeshwanth trial 2 | retrain → **refused**: p = 0.65 ≥ 0.05 and macro-F1 0.24 < champion 0.56 |

What this shows: the loop, triggers, quarantine and gate work and protect the served model. What it does **not** show: that retraining improves accuracy. Only one arrival could be scored by a champion before training (1/1 correct), which is statistically meaningless. Known gate limitation: it compares macro-F1 across models evaluated on different cohorts, so it is only approximately like-for-like.
The drift-alarm retraining trigger is implemented but cannot fire with a 7-sample reference (never armed); it is covered by monitor unit tests, not an end-to-end test.

## 6. Throughput and latency (Docker, Locust, real 2–3 s clip incl. upload, 45 s per run)

| Workers | Users | Requests | Failures | Req/s | Median | p95 |
|---|---|---|---|---|---|---|
| 1 | 1 | 21 | 0 | 0.48 | 2.1 s | 2.1 s |
| 1 | 4 | 52 | 0 | 1.21 | 3.3 s | 3.5 s |
| 2 | 4 | 48 | 0 | 1.14 | 3.5 s | 3.6 s |
| 4 | 4 | 44 | 0 | 1.04 | 3.8 s | 4.2 s |
| 4 | 8 | 49 | 0 | 1.11 | 6.5 s | 7.0 s |

Zero failures in 214 requests. **Throughput saturates at ≈ 1.1–1.2 clips/s (≈ 70/min) on this 10-core node regardless of worker count**: one request already uses several cores inside MediaPipe, so extra workers add queueing, not capacity. Latency grows linearly with concurrency. Scaling beyond this needs more nodes; that was **not** tested.

## 7. Availability and fault tolerance

| Test | Result |
|---|---|
| Steady-state probe (`/health` every 5 s, no faults) | **60/60 OK = 100 % over 5 min**; mean 5.9 ms. A 5-minute window cannot support any claim near 99.5 %. |
| Kill 1 of 2 API workers under a 100 ms probe (3 trials) | 0 failed probes; replacement worker up in 0.03 s (verified by PID) |
| SIGKILL the uvicorn master → container exits (3 trials) | Docker restart policy: healthy again in 0.9–1.0 s; restart counter = 3 |
| `docker restart` (3 trials) | healthy in 3.8–3.9 s |
| `docker kill` | **Not restarted**: Docker treats it as a manual stop (`restart: always` does not apply) |

To collect a defensible availability figure, leave `python scripts/uptime_probe.py --duration <seconds>` running from another machine until submission and quote the measured window.

## 8. Not done / caveats

- No on-phone (Android/iOS) latency; no Prometheus server or Grafana dashboard (the `/metrics` endpoint exists; counters are per worker process).
- The GitHub Actions workflow is untested until pushed. The Streamlit client was edited and syntax-checked but not launched. The old Streamlit Cloud deployment would need the API hosted somewhere.
- Quality-gate thresholds were set after inspecting these 9 recordings (e.g. `min_fps` 50 because phones record 58–60 fps variable-rate video); they need re-validation on new data.
- Pose numbers differ slightly between macOS and the Linux container (e.g. Amith 8.30 vs 8.32 m/s), as expected for different numerical back-ends; pin the platform when quoting results.
- Environment note: `protobuf==3.20.3` is required by MediaPipe 0.10.9 and TensorFlow; MLflow prints a harmless dependency warning for it.
