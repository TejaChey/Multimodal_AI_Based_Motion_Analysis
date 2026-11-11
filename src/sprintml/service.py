"""FastAPI inference service.

POST /predict   video + accelerometer CSV + gyroscope CSV -> tier, features, quality report
GET  /health    liveness + loaded model
GET  /metrics   Prometheus metrics (per-stage latency, request/error counts)
GET  /drift     current drift report (monitor)
GET  /model     champion metadata
POST /admin/reload   re-read the champion (it is also hot-reloaded when promoted)
"""
import ctypes
import gc
import json
import os
import signal
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest, multiprocess

from . import __version__
from .config import CLASS_NAMES, art_dir, load_params
from .features import extract_features
from .imu import load_imu
from .labels import assign_tiers
from .monitor import drift_report, load_reference, write_report
from .pose import extract_pose
from .quality import check_recording
from .store import FeatureStore

app = FastAPI(title="Sprint biomechanics API", version=__version__)
P = load_params()
STAGE = Histogram("sprintml_stage_seconds", "Latency per pipeline stage", ["stage"],
                  buckets=(.01, .05, .1, .25, .5, 1, 2, 3, 5, 10, 20))
REQS = Counter("sprintml_requests_total", "Prediction requests", ["status"])
MODEL_V = Gauge("sprintml_model_version", "Loaded champion version", multiprocess_mode="max")
INFLIGHT = Gauge("sprintml_inflight_requests", "Requests being processed", multiprocess_mode="livesum")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 2))     # simultaneous pose pipelines per process (memory bound)
QUEUE_TIMEOUT = float(os.environ.get("QUEUE_TIMEOUT_S", 20))  # how long a request may wait for a slot before being shed
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
RECYCLE_RSS_MB = float(os.environ.get("RECYCLE_RSS_MB", 0))   # >0: gracefully restart the process above this RSS
_draining = threading.Event()
_active = 0
_lock = threading.Lock()
_state = {"bundle": None, "meta": None, "mtime": None}
_store = None


def _champion_path() -> Path:
    return art_dir() / "champion"


def load_champion() -> bool:
    d = _champion_path()
    if not (d / "model.joblib").exists():
        return False
    with _lock:
        _state["bundle"] = joblib.load(d / "model.joblib")
        _state["meta"] = json.loads((d / "meta.json").read_text())
        _state["mtime"] = (d / "meta.json").stat().st_mtime
        MODEL_V.set(float(_state["meta"].get("version", 0) or 0))
    return True


def _maybe_reload():
    """Hot-reload when a retraining job promotes a new champion."""
    m = _champion_path() / "meta.json"
    if m.exists() and m.stat().st_mtime != _state["mtime"]:
        load_champion()


def store() -> FeatureStore:
    global _store
    if _store is None:
        _store = FeatureStore()
    return _store


@app.on_event("startup")
def _startup():
    load_champion()


def _rss_mb() -> float:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except OSError:
        return 0.0


def _exit_when_idle():
    while _active > 0:
        time.sleep(0.2)
    os.kill(os.getpid(), signal.SIGTERM)      # uvicorn drains and exits; the supervisor / orchestrator restarts it


def _maybe_recycle():
    """Memory safety net: above RECYCLE_RSS_MB stop taking traffic (health -> 503), finish in-flight work, exit."""
    if RECYCLE_RSS_MB and not _draining.is_set() and _rss_mb() > RECYCLE_RSS_MB:
        _draining.set()
        threading.Thread(target=_exit_when_idle, daemon=True).start()


def _release_memory():
    """Return freed memory to the OS: MediaPipe/OpenCV buffers otherwise fragment the glibc heap (~12 MiB/request)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:       # not glibc (e.g. macOS dev run)
        pass


@contextmanager
def timed(stage, bucket):
    t = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t
        STAGE.labels(stage).observe(dt)
        bucket[stage] = round(dt * 1000, 1)


def _save(upload: UploadFile, td: Path, name: str) -> Path:
    path = td / name
    with open(path, "wb") as f:
        while chunk := upload.file.read(1 << 20):
            f.write(chunk)
    return path


@app.get("/health")
async def health():       # async: served on the event loop, so it stays responsive when the worker threads are busy
    _maybe_reload()
    ok = _state["bundle"] is not None and not _draining.is_set()
    body = {"status": "ok" if ok else ("draining" if _draining.is_set() else "no_model"), "model_version": (_state["meta"] or {}).get("version"),
            "model_validated": (_state["meta"] or {}).get("validated"), "service_version": __version__}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/model")
def model_info():
    if _state["meta"] is None:
        raise HTTPException(503, "no model loaded")
    return _state["meta"]


@app.get("/reference")
def reference():
    """Per-tier feature means of the reference set (descriptive comparison, not a validated norm)."""
    try:
        ref = load_reference()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    means = ref.groupby("label")[list(ref.columns[4:])].mean()
    return {"n": int(len(ref)), "tier_means": {CLASS_NAMES[int(k)]: v.round(4).to_dict() for k, v in means.iterrows()},
            "tier_counts": {CLASS_NAMES[int(k)]: int(v) for k, v in ref.label.value_counts().items()}}


@app.post("/admin/reload")
def reload_model():
    return {"reloaded": load_champion(), "model_version": (_state["meta"] or {}).get("version")}


@app.get("/metrics")
def metrics():
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):      # aggregate across uvicorn worker processes
        reg = CollectorRegistry()
        multiprocess.MultiProcessCollector(reg)
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/drift")
def drift():
    win = store().frame(last=P["monitor"]["window"])
    if win.empty:
        return {"status": "no_requests_yet"}
    try:
        rep = drift_report(load_reference(), win, P)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    write_report(rep)
    return rep


@app.post("/predict")
def predict(video: UploadFile = File(...), accel: UploadFile = File(...), gyro: UploadFile = File(...),
            athlete_id: str = Form("anonymous"), recording_id: str = Form(None)):
    _maybe_reload()
    if _state["bundle"] is None:
        REQS.labels("no_model").inc()
        raise HTTPException(503, "no model loaded")
    rid = recording_id or f"req-{uuid.uuid4().hex[:10]}"
    timings, t_all = {}, time.perf_counter()
    if not _slots.acquire(timeout=QUEUE_TIMEOUT):          # load shedding instead of exhausting memory
        REQS.labels("shed").inc()
        raise HTTPException(503, "server busy, retry shortly", headers={"Retry-After": "5"})
    if _draining.is_set():                                   # recycling: let the load balancer retry elsewhere
        _slots.release()
        REQS.labels("shed").inc()
        raise HTTPException(503, "instance restarting, retry", headers={"Retry-After": "2"})
    global _active
    with _lock:
        _active += 1
    INFLIGHT.inc()
    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            v, a, g = _save(video, td, "v.mp4"), _save(accel, td, "a.csv"), _save(gyro, td, "g.csv")
            try:
                with timed("pose", timings):
                    pose = extract_pose(v, P)
                with timed("features", timings):
                    x = extract_features(pose, load_imu(a, g), P)
            except ValueError as e:
                REQS.labels("unprocessable").inc()
                raise HTTPException(422, f"could not process recording: {e}")
        feats, meta = x["features"], x["meta"]
        q = check_recording(feats, meta, P)
        if not q["passed"]:
            REQS.labels("quality_rejected").inc()
            store().add(rid, athlete_id, feats, q, meta["v_statures_per_s"], timings=timings)
            return JSONResponse({"recording_id": rid, "prediction": None, "quality": q,
                                 "detail": "recording failed the data-quality gate: " + ", ".join(q["failed"])},
                                status_code=422)
        with timed("model", timings):
            b = _state["bundle"]
            X = b["scaler"].transform(pd.DataFrame([feats])[b["features"]].values)
            proba = b["clf"].predict_proba(X)[0]
            tier = CLASS_NAMES[int(b["clf"].classes_[int(np.argmax(proba))])]
        timings["total"] = round((time.perf_counter() - t_all) * 1000, 1)
        STAGE.labels("total").observe(timings["total"] / 1000)
        m = _state["meta"]
        thr = m.get("tier_thresholds_statures_per_s")
        store().add(rid, athlete_id, feats, q, meta["v_statures_per_s"], model_version=m.get("version"),
                    predicted_tier=tier, timings=timings)
        REQS.labels("ok").inc()
        return {"recording_id": rid, "athlete_id": athlete_id, "prediction": tier,
                "probabilities": {CLASS_NAMES[int(c)]: round(float(pr), 4) for c, pr in zip(b["clf"].classes_, proba)},
                "model_version": m.get("version"),
                "model_status": "validated" if m.get("validated") else
                f"PROVISIONAL: not statistically above chance (LOSO accuracy {m['loso_accuracy']:.0%}, "
                f"p={m['perm_p']:.2f}, n={m['n_recordings']}). Use the measured velocity, not this tier.",
                "measured_velocity_mps": round(meta["v_mps"], 3),
                "velocity_tier": CLASS_NAMES[int(assign_tiers([meta["v_statures_per_s"]], thr)[0])] if thr else None,
                "features": {k: round(float(v), 4) for k, v in feats.items()},
                "quality": q, "timings_ms": timings}
    finally:
        INFLIGHT.dec()
        _release_memory()
        with _lock:
            _active -= 1
        _maybe_recycle()
        _slots.release()
