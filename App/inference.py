"""Thin client of the sprintml API (the model, pose extraction and monitoring live in the service)."""
import os
from pathlib import Path

import requests

API = os.environ.get("SPRINTML_API_URL", "http://localhost:8000")


def run_inference(video_path: Path, accel_path: Path, gyro_path: Path):
    """Returns (velocity_tier, metrics, error). metrics also carries the model's (provisional) tier."""
    try:
        with open(video_path, "rb") as v, open(accel_path, "rb") as a, open(gyro_path, "rb") as g:
            r = requests.post(f"{API}/predict", files={"video": v, "accel": a, "gyro": g}, timeout=300)
    except Exception as e:
        return None, None, f"Could not reach the analysis service at {API}: {e}"
    if r.status_code == 422:
        body = r.json()
        why = body.get("detail") or body
        return None, None, f"This recording was rejected by the data-quality gate: {why}"
    if r.status_code != 200:
        return None, None, f"Service error {r.status_code}: {r.text[:300]}"
    d = r.json()
    metrics = {**d["features"], "measured_velocity_mps": d["measured_velocity_mps"], "model_tier": d["prediction"],
               "model_status": d["model_status"], "timings_ms": d["timings_ms"]}
    return d["velocity_tier"], metrics, ""


def fetch_reference():
    try:
        return requests.get(f"{API}/reference", timeout=10).json()
    except Exception:
        return None
