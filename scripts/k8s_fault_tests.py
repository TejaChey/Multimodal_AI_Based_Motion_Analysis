"""Kubernetes fault tests under continuous traffic (kind cluster, NodePort on localhost:8080).

A. rolling update (`kubectl rollout restart`): /health probed every 50 ms + one client sending real /predict requests.
B. pod crash (`kubectl delete pod --grace-period=0 --force`): same traffic; time until 2 replicas are Ready again.
Counts failed requests; reports totals.
"""
import json, subprocess, threading, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "Phase_2/data/sprint_raw/Tejas"
URL = "http://localhost:8080"; sh = lambda *a: subprocess.run(a, capture_output=True, text=True).stdout.strip()

def multipart():
    b = "----b" + str(time.time_ns()); parts = []
    for name, p, ct in (("video", next(D.glob("*.mp4")), "video/mp4"), ("accel", D / "WatchAccelerometer.csv", "text/csv"), ("gyro", D / "WatchGyroscope.csv", "text/csv")):
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{name}"; filename="{p.name}"\r\nContent-Type: {ct}\r\n\r\n'.encode() + p.read_bytes() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode()); return b"".join(parts), f"multipart/form-data; boundary={b}"
BODY, CT = multipart()

def ready(): return int((sh("kubectl", "get", "deploy", "sprintml-api", "-o", "jsonpath={.status.readyReplicas}") or 0))
WANT = 2

def run(action, label):
    stop, h, p = threading.Event(), [], []
    def health():
        while not stop.is_set():
            try:
                with urllib.request.urlopen(URL + "/health", timeout=2) as r: h.append(r.status == 200)
            except Exception: h.append(False)
            time.sleep(0.05)
    def predict():
        while not stop.is_set():
            try:
                req = urllib.request.Request(URL + "/predict", data=BODY, headers={"Content-Type": CT})
                with urllib.request.urlopen(req, timeout=60) as r: p.append(r.status == 200)
            except Exception: p.append(False)
    ts = [threading.Thread(target=health), threading.Thread(target=predict)]
    [t.start() for t in ts]; time.sleep(5); t0 = time.time(); action()
    while time.time() - t0 < 180:
        if ready() == 2 and sh("kubectl", "rollout", "status", "deploy/sprintml-api", "--timeout=1s").startswith("deployment"): break
        time.sleep(0.5)
    recovered = time.time() - t0; time.sleep(8); stop.set(); [t.join() for t in ts]
    r = {"test": label, "health_probes": len(h), "health_failed": h.count(False), "predict_requests": len(p),
         "predict_failed": p.count(False), "seconds_until_2_ready": round(recovered, 1)}; print(json.dumps(r), flush=True); return r

def roll(): sh("kubectl", "rollout", "restart", "deployment/sprintml-api")
def crash():
    pod = sh("kubectl", "get", "pods", "-l", "app=sprintml-api", "-o", "jsonpath={.items[0].metadata.name}")
    sh("kubectl", "delete", "pod", pod, "--grace-period=0", "--force")
res = [run(roll, "rolling_update"), run(roll, "rolling_update"), run(crash, "pod_crash"), run(crash, "pod_crash")]
(ROOT / "artifacts" / "k8s_fault_results.json").write_text(json.dumps(res, indent=2))
