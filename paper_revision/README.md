# Camera-ready revision: ICICF-2027 Paper 768

Everything here was produced from the **existing** 9 recordings. No new data was collected.

## What's in this folder

| File | Use |
|---|---|
| `RESPONSE_TO_REVIEWERS.md` | Point-by-point reply to the 10 comments. No `[CONFIRM]` items remain. |
| `revised_sections.tex` | Drop-in LaTeX for the abstract, contributions, methods, results, latency, drift, limitations, reproducibility and conclusion |
| `figures/` | New figures: speed labels, sync validation, confusion matrices, drift power |
| `MLOPS_RESULTS.md` | What the MLOps system does, with every measured number (load, faults, drift replay, retraining replay) |
| `MLOPS_PLAN.md` | The plan, with implementation status at the bottom |
| `CLAIMS_AUDIT.md` | The original audit of the draft's claims against the repo |
| `code/run_all.sh` | Regenerates every number from the raw recordings (about 25 min on an M5) |
| `requirements-lock.txt` | Exact package versions |
| `results/` | All CSV/JSON outputs plus `sprint_mlp.tflite` |

## What changed in the analysis

1. **Labels:** K-Means on the input features (circular) → measured running velocity (objective, automatic).
2. **Leakage:** augmentation used to run before the split → now runs inside each Leave-One-Subject-Out fold, on training athletes only.
3. **Features:** stride length (= 15 m / steps, which was wrong and leaks the label), clip duration and frame count removed. Ground contact is now detected from the stance foot being stationary. Static false pose detections are filtered out.
4. **New validations:** video↔watch cadence agreement, clock consistency, measured latency, KS power analysis, device-shift and chronological retraining tests.

## Headline numbers (honest)

| | Result |
|---|---|
| Velocity labels | 6.7–9.4 m/s, fit R² ≥ 0.998, both Yeshwanth trials in the same tier |
| Video↔watch cadence (n = 7) | MAE 0.073 Hz, r = 0.95 |
| Mismatched watch files found | Chandra1 (−102 s), Jeeva (−422 s) |
| Best LOSO accuracy | 55.6% IMU-only (chance 33.3%, p = 0.16). **No model is significantly above chance; fusion does not help.** |
| Latency (M5) | pose 11.5 ms/frame, 2.1 s per clip, TFLite classifier ~1 µs |
| Drift monitor | needs ≥ 100 reference recordings and windows ≥ 30 to be reliable |
| Retraining | no measurable benefit at this sample size (2/5 vs 2/5) |

**82% accuracy, +12% fusion gain, 87% on UTD-MHAD, 95 ms mobile latency and 99.5% availability are not supported and have been removed or replaced.** The paper now presents a validated measurement and MLOps pipeline plus an honest pilot result.

## Still to do (authors)

- [x] Watch = Galaxy Watch4 Classic, left wrist; consent taken; participant details kept confidential (done). Do **not** claim ethics-committee approval or *signed* forms unless that is true.
- [ ] Fix citation numbering. Most in-text numbers point to the wrong references; check that each reference exists.
- [ ] Delete the garbled paragraph in §V-B ("ESM-COTS … CMA project"), the empty §II-D, and the "Tap to enter" overlay in Fig. 2. Replace Fig. 3 (it shows "96.67%" and the K-Means autograder) and Fig. 6 (100% on leaked data).
- [ ] Optional: ask the TPC whether a small title change is allowed, e.g. "An MLOps Pipeline for Smartphone-Based Sprint Biomechanics: From Research Prototype Toward a Coaching Platform". "Production-ready" and "scalable" are not demonstrated.
- [ ] Optional: a teammate runs `python code/06_latency.py <device-name>` on their laptop to add a second device column to the latency table.
- [x] Repository URL added; raw data stated as not releasable (consent).
- [ ] **Push/merge the `mlops-camera-ready` branch to that repository** so the MLOps code the paper cites is actually there.

## Reproduce

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r paper_revision/requirements-lock.txt
bash paper_revision/code/run_all.sh
```


## MLOps update (2026-11-10)

The MLOps system now exists in `src/sprintml/` (DVC pipeline, MLflow registry with a promotion gate, FastAPI + Docker,
drift monitor, retraining loop, 43 tests, CI workflow). Measured evidence: `MLOPS_RESULTS.md`. Run `dvc repro`, `pytest`,
`docker compose up --build` from the repository root. The reviewer response and `revised_sections.tex` are updated
to match. Still open: on-phone latency, running the CI on GitHub, a longer uptime probe, and the repository URL.
