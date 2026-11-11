"""Summarise a uptime_probe CSV: availability, longest outage, window. python scripts/uptime_summary.py artifacts/uptime_long.csv"""
import csv, sys
from datetime import datetime
rows = list(csv.DictReader(open(sys.argv[1])))
ok = [int(r["ok"]) for r in rows]; t = [datetime.fromisoformat(r["utc"]) for r in rows]
run = best = 0
for v in ok:
    run = 0 if v else run + 1; best = max(best, run)
span = (t[-1] - t[0]).total_seconds()
import math
n, up = len(ok), sum(ok); z = 1.96; p = up / n
lo = (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
print({"probes": n, "up": up, "availability_pct": round(100*p, 3), "wilson95_lower_pct": round(100*lo, 3),
       "window_hours": round(span/3600, 2), "longest_outage_probes": best})
