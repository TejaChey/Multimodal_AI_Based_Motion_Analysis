"""Load test: realistic /predict requests (a real 2-3 s sprint clip + watch CSVs).

    locust -f scripts/locustfile.py --headless -u 4 -r 4 -t 60s --host http://localhost:8000
Set LOCUST_DUMP=path.json to also write per-request outcomes (status, seconds) so served and shed requests
can be analysed separately.
"""
import json
import os
import time
from pathlib import Path

from locust import HttpUser, constant, events, task

D = Path(os.environ.get("CLIP_DIR", "Phase_2/data/sprint_raw/Tejas"))
VIDEO = next(D.glob("*.mp4")).read_bytes()
ACC, GYR = (D / "WatchAccelerometer.csv").read_bytes(), (D / "WatchGyroscope.csv").read_bytes()
OUTCOMES = []


class Athlete(HttpUser):
    wait_time = constant(0)

    @task(1)
    def predict(self):
        t0 = time.time()
        with self.client.post("/predict", files={"video": ("v.mp4", VIDEO, "video/mp4"), "accel": ("a.csv", ACC, "text/csv"),
                                                 "gyro": ("g.csv", GYR, "text/csv")}, data={"athlete_id": "load"},
                              name="/predict", catch_response=True) as r:
            OUTCOMES.append((r.status_code, round(time.time() - t0, 3)))
            if r.status_code == 503:
                r.failure("shed (503)")


@events.quitting.add_listener
def dump(environment, **kw):
    if os.environ.get("LOCUST_DUMP"):
        Path(os.environ["LOCUST_DUMP"]).write_text(json.dumps(OUTCOMES))
