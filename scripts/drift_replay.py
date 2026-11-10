"""Replay real recordings with simulated device shifts through the real API code path (in-process).

Scenarios: baseline | watch downsampled to 50 Hz | watch accelerometer gain x1.5.
Shows which layer catches each shift: the data-quality gate or the drift monitor.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from fastapi.testclient import TestClient

from sprintml import service
from sprintml.config import raw_dir

REAL = ROOT / "artifacts"
CLEAN = [c for c, q in json.loads((REAL / "quality_report.json").read_text()).items() if q["passed"]]


def csv(df):
    return io.BytesIO(df.to_csv(index=False).encode())


def scenario(name, accel_fn=lambda a: a, gyro_fn=lambda g: g):
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(REAL / "champion", Path(tmp) / "champion")
        os.environ["SPRINTML_ARTIFACTS"] = tmp
        service._store, service._state["mtime"] = None, None
        out = {"scenario": name, "sent": 0, "ok": 0, "quality_rejected": 0, "rejected_for": {}}
        with TestClient(service.app) as c:
            for clip in CLEAN:
                d = raw_dir() / clip
                a, g = pd.read_csv(d / "WatchAccelerometer.csv"), pd.read_csv(d / "WatchGyroscope.csv")
                vid = next(d.glob("*.mp4"))
                r = c.post("/predict", files={"video": (vid.name, open(vid, "rb"), "video/mp4"),
                                              "accel": ("a.csv", csv(accel_fn(a))), "gyro": ("g.csv", csv(gyro_fn(g)))},
                           data={"athlete_id": clip, "recording_id": f"{name}-{clip}"})
                out["sent"] += 1
                if r.status_code == 200:
                    out["ok"] += 1
                elif r.status_code == 422 and r.json().get("quality"):
                    out["quality_rejected"] += 1
                    for f in r.json()["quality"]["failed"]:
                        out["rejected_for"][f] = out["rejected_for"].get(f, 0) + 1
            rep = c.get("/drift").json()
            out["drift_status"] = rep.get("status")
            out["armed"] = rep.get("armed")
            out["n_window"] = rep.get("n_window")
            out["drifted_features"] = rep.get("drifted_features", [])
            out["top_features_p"] = [(f["feature"], round(f["p_value"], 5)) for f in rep.get("features", [])[:3]]
        os.environ.pop("SPRINTML_ARTIFACTS")
        print(json.dumps(out), flush=True)
        return out


def gain(a):
    a = a.copy()
    a[["x", "y", "z"]] *= 1.5
    return a


res = [scenario("baseline"),
       scenario("watch_50Hz", accel_fn=lambda a: a.iloc[::2], gyro_fn=lambda g: g.iloc[::2]),
       scenario("accel_gain_x1.5", accel_fn=gain)]
(REAL / "drift_replay_results.json").write_text(json.dumps(res, indent=2))
