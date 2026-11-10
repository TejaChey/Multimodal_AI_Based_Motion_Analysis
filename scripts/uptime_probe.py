"""External availability probe: GET /health every INTERVAL seconds; writes a CSV and prints availability.

    python scripts/uptime_probe.py --url http://localhost:8000/health --interval 5 --duration 3600
Leave it running (e.g. on another machine) until submission and report the measured window honestly.
"""
import argparse
import csv
import json
import time
import urllib.request
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000/health")
ap.add_argument("--interval", type=float, default=5)
ap.add_argument("--duration", type=float, default=300)
ap.add_argument("--out", default="artifacts/uptime_probe.csv")
a = ap.parse_args()
rows, t_end = [], time.time() + a.duration
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["utc", "ok", "latency_ms", "status"])
    while time.time() < t_end:
        t0, ok, status = time.time(), 0, "error"
        try:
            with urllib.request.urlopen(a.url, timeout=3) as r:
                status, ok = r.status, int(r.status == 200)
        except Exception as e:
            status = type(e).__name__
        lat = round((time.time() - t0) * 1000, 1)
        rows.append(ok)
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), ok, lat, status])
        f.flush()
        time.sleep(max(0, a.interval - (time.time() - t0)))
n, up = len(rows), sum(rows)
print(json.dumps({"probes": n, "up": up, "availability_pct": round(100 * up / n, 3),
                  "window_s": round(a.duration), "interval_s": a.interval}))
