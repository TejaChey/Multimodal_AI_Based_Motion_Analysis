"""API contract tests (pose extraction is replaced by a synthetic runner; everything else is real)."""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import synthetic_imu, synthetic_pose, synthetic_table
from sprintml import service, train
from sprintml.labels import assign_tiers, tier_thresholds


def csv_bytes(df):
    return df.to_csv(index=False).encode()


@pytest.fixture
def client(tmp_path, monkeypatch, params):
    monkeypatch.setenv("SPRINTML_ARTIFACTS", str(tmp_path))
    monkeypatch.setattr(service, "extract_pose", lambda path, p: synthetic_pose())
    monkeypatch.setattr(service, "_store", None)
    service._state.update(bundle=None, meta=None, mtime=None)
    with TestClient(service.app) as c:
        yield c


def train_champion(params):
    df = synthetic_table(3, seed=1)
    df["label"] = assign_tiers(df.v_statures_per_s, tier_thresholds(df.v_statures_per_s))
    return train.train_and_register(df, params, "test", n_perm=30)


def files(imu_hz=100.0):
    a, g = synthetic_imu(hz=imu_hz)
    return {"video": ("v.mp4", io.BytesIO(b"not-a-real-video"), "video/mp4"),
            "accel": ("a.csv", io.BytesIO(csv_bytes(a)), "text/csv"),
            "gyro": ("g.csv", io.BytesIO(csv_bytes(g)), "text/csv")}


def test_no_model_returns_503(client):
    assert client.get("/health").status_code == 503
    assert client.post("/predict", files=files()).status_code == 503


def test_health_predict_metrics_contract(client, params):
    train_champion(params)
    assert client.post("/admin/reload").json()["reloaded"]
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["model_version"] == "1"

    r = client.post("/predict", files=files(), data={"athlete_id": "S9", "recording_id": "t1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in ("Beginner", "Intermediate", "Elite")
    assert pytest.approx(sum(body["probabilities"].values()), abs=1e-3) == 1
    assert body["measured_velocity_mps"] == pytest.approx(4.5 * 1.70, rel=0.05)
    assert body["quality"]["passed"] and len(body["features"]) == 19
    assert {"pose", "features", "model", "total"} <= set(body["timings_ms"])
    assert "model_status" in body

    m = client.get("/metrics").text
    assert 'sprintml_requests_total{status="ok"} 1.0' in m and 'sprintml_stage_seconds_count{stage="pose"}' in m
    assert client.get("/drift").json()["status"] in ("insufficient_data", "informational_alarm_unarmed", "ok", "ALARM")


def test_quality_failure_is_rejected_and_logged(client, params):
    train_champion(params)
    client.post("/admin/reload")
    r = client.post("/predict", files=files(imu_hz=30.0))          # watch recorded at 30 Hz
    assert r.status_code == 422 and r.json()["prediction"] is None
    assert "imu_rate_hz" in r.json()["quality"]["failed"]
    assert service.store().count() == 1 and service.store().count(passed_only=True) == 0


def test_malformed_input_is_a_clean_422(client, params):
    train_champion(params)
    client.post("/admin/reload")
    bad = files()
    bad["accel"] = ("a.csv", io.BytesIO(b"nonsense,columns\n1,2\n"), "text/csv")
    assert client.post("/predict", files=bad).status_code == 422
    assert client.post("/predict").status_code == 422               # missing parts


def test_hot_reload_on_promotion(client, params):
    train_champion(params)
    client.post("/admin/reload")
    first = client.get("/health").json()["model_version"]
    # a retraining job exports a new champion; the service picks it up without a restart
    import json, os
    from sprintml.config import art_dir
    meta = json.loads((art_dir() / "champion" / "meta.json").read_text())
    meta["version"] = "2"
    p = art_dir() / "champion" / "meta.json"
    p.write_text(json.dumps(meta))
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 5))
    assert first == "1" and client.get("/health").json()["model_version"] == "2"


def test_overload_is_shed_not_crashed(client, params, monkeypatch):
    """With one slot and a short queue timeout, concurrent requests beyond capacity get a fast 503 + Retry-After."""
    import threading, time
    train_champion(params)
    client.post("/admin/reload")
    monkeypatch.setattr(service, "_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(service, "QUEUE_TIMEOUT", 0.2)
    slow = lambda path, p: (time.sleep(0.8), synthetic_pose())[1]
    monkeypatch.setattr(service, "extract_pose", slow)
    codes = []
    def go():
        codes.append(client.post("/predict", files=files()).status_code)
    ts = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in ts]
    # health must stay responsive while predictions are busy
    t0 = time.time(); assert client.get("/health").status_code == 200; assert time.time() - t0 < 0.5
    [t.join() for t in ts]
    assert 200 in codes and codes.count(503) >= 2 and set(codes) <= {200, 503}
    assert 'sprintml_requests_total{status="shed"}' in client.get("/metrics").text


def test_memory_recycle_drains_then_exits(client, params, monkeypatch):
    """Above the RSS threshold the process stops advertising health and asks to be restarted after in-flight work."""
    import signal
    train_champion(params)
    client.post("/admin/reload")
    killed = []
    monkeypatch.setattr(service, "RECYCLE_RSS_MB", 10.0)
    monkeypatch.setattr(service, "_rss_mb", lambda: 999.0)
    monkeypatch.setattr(service.os, "kill", lambda pid, sig: killed.append(sig))
    service._draining.clear()
    assert client.post("/predict", files=files()).status_code == 200      # in-flight request still completes
    import time
    for _ in range(30):
        if killed:
            break
        time.sleep(0.1)
    assert killed == [signal.SIGTERM]
    h = client.get("/health")
    assert h.status_code == 503 and h.json()["status"] == "draining"
    service._draining.clear()
