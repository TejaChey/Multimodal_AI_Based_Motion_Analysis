#!/usr/bin/env bash
# Reproduce every number in the camera-ready revision.
# Requires Python 3.11 and ffprobe (FFmpeg) on PATH.
#   python3.11 -m venv .venv && .venv/bin/pip install -r paper_revision/requirements-lock.txt
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}
$PY 01_extract_pose.py          # MediaPipe Pose landmarks per frame
$PY 02_speed_labels.py          # objective velocity-based tiers
$PY 03_features.py              # corrected video + IMU features
$PY 04_sync_validation.py       # video <-> watch integration checks
N_PERM=${N_PERM:-200} $PY 05_evaluate.py   # LOSO classification + regression
$PY 06_latency.py all_cores     # latency on this machine
if [[ "$(uname)" == "Darwin" ]]; then taskpolicy -b $PY 06_latency.py efficiency_cores; fi
$PY 07_drift_retraining.py      # KS thresholds, device shift, chronological retraining
