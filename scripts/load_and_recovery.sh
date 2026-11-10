#!/usr/bin/env bash
# Throughput / latency vs workers and users (Locust), then crash-recovery time. Uses a COPY of the champion.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOC=.venv/bin/locust
LT=/tmp/lt_art; rm -rf $LT; mkdir -p $LT; cp -r artifacts/champion $LT/champion
OUT=artifacts/load_test; mkdir -p $OUT; echo "workers,users,requests,failures,rps,median_ms,p95_ms,p99_ms" > $OUT/summary.csv
wait_up() { for i in $(seq 1 60); do curl -fs localhost:8000/health >/dev/null && return 0; sleep 1; done; return 1; }
for cfg in "1 1" "1 4" "2 4" "4 4" "4 8"; do
  set -- $cfg; W=$1; U=$2
  docker rm -f lt >/dev/null 2>&1
  docker run -d --name lt -p 8000:8000 -e WORKERS=$W -v $LT:/data sprintml-api:latest >/dev/null
  wait_up || { echo "no start W=$W"; continue; }
  R=Phase_2/data/sprint_raw/Tejas
  for k in $(seq 1 $W); do curl -s -o /dev/null -X POST localhost:8000/predict -F "video=@$R/Tejas.mp4" -F "accel=@$R/WatchAccelerometer.csv" -F "gyro=@$R/WatchGyroscope.csv"; done   # warm each worker
  $LOC -f scripts/locustfile.py --headless -u $U -r $U -t 45s --host http://localhost:8000 --csv $OUT/w${W}_u${U} --only-summary >/dev/null 2>&1
  $PY - <<PYEOF >> $OUT/summary.csv
import csv
r = [x for x in csv.DictReader(open("$OUT/w${W}_u${U}_stats.csv")) if x["Name"] == "/predict"][0]
print(",".join(map(str, [$W, $U, r["Request Count"], r["Failure Count"], round(float(r["Requests/s"]), 2),
      r["Median Response Time"], r["95%"], r["99%"]])))
PYEOF
  echo "W=$W U=$U done"; docker rm -f lt >/dev/null 2>&1
done
# crash recovery: kill the main process, restart policy brings it back
echo "trial,recovery_s" > $OUT/recovery.csv
docker run -d --name lt --restart always -p 8000:8000 -e WORKERS=2 -v $LT:/data sprintml-api:latest >/dev/null
wait_up
for t in 1 2 3; do
  S=$($PY -c "import time;print(time.time())")
  docker kill lt >/dev/null
  until curl -fs localhost:8000/health >/dev/null; do sleep 0.25; done
  E=$($PY -c "import time;print(time.time())")
  $PY -c "print('$t,%.1f' % ($E-$S))" >> $OUT/recovery.csv
  sleep 3
done
docker rm -f lt >/dev/null 2>&1
echo ALLDONE
