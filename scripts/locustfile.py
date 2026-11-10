"""Load test: realistic /predict requests (a real 2-3 s sprint clip + watch CSVs).

    locust -f scripts/locustfile.py --headless -u 4 -r 4 -t 60s --host http://localhost:8000
"""
import os
from pathlib import Path

from locust import HttpUser, constant, task

D = Path(os.environ.get("CLIP_DIR", "Phase_2/data/sprint_raw/Tejas"))
VIDEO = next(D.glob("*.mp4")).read_bytes()
ACC, GYR = (D / "WatchAccelerometer.csv").read_bytes(), (D / "WatchGyroscope.csv").read_bytes()


class Athlete(HttpUser):
    wait_time = constant(0)

    @task(1)
    def predict(self):
        self.client.post("/predict", files={"video": ("v.mp4", VIDEO, "video/mp4"), "accel": ("a.csv", ACC, "text/csv"),
                                            "gyro": ("g.csv", GYR, "text/csv")}, data={"athlete_id": "load"},
                         name="/predict")
