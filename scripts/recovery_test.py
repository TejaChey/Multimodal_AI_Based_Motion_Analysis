"""Fault tests against the Docker image (uses a COPY of the champion):

A. worker crash: SIGKILL one of 2 uvicorn workers while /health is probed every 100 ms.
   Reports failed probes, and seconds until 2 workers are running again (supervisor respawn).
B. container restart (`docker restart`): seconds until /health is 200 again.
C. process crash: SIGKILL the uvicorn master (container exits non-zero) -> Docker restart policy revives it.
(`docker kill` is treated by Docker as a manual stop, so the restart policy does not revive it.)
"""
import json
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LT = Path("/tmp/lt_art")
shutil.rmtree(LT, ignore_errors=True)
LT.mkdir()
shutil.copytree(ROOT / "artifacts" / "champion", LT / "champion")
sh = lambda *a: subprocess.run(a, capture_output=True, text=True).stdout.strip()


def healthy():
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def wait_healthy(limit=90):
    t = time.time()
    while time.time() - t < limit:
        if healthy():
            return time.time() - t
        time.sleep(0.1)
    return float("nan")


COUNT = ("import os;print(sum(1 for p in os.listdir('/proc') if p.isdigit() and "
         "b'spawn_main' in open(f'/proc/{p}/cmdline','rb').read()))")
KILL = ("import os,signal;ps=[int(p) for p in os.listdir('/proc') if p.isdigit() and "
        "b'spawn_main' in open(f'/proc/{p}/cmdline','rb').read()];os.kill(ps[0],signal.SIGKILL)")

sh("docker", "rm", "-f", "lt")
sh("docker", "run", "-d", "--name", "lt", "--restart", "always", "-p", "8000:8000", "-e", "WORKERS=2",
   "-v", f"{LT}:/data", "sprintml-api:latest")
print("startup to healthy: %.1f s" % wait_healthy(), flush=True)
time.sleep(5)
res = {"worker_crash": [], "container_restart_s": [], "startup_s": None}
for trial in range(3):
    stop, log = threading.Event(), []

    def probe():
        while not stop.is_set():
            log.append(healthy())
            time.sleep(0.1)

    th = threading.Thread(target=probe)
    th.start()
    time.sleep(2)
    t0 = time.time()
    sh("docker", "exec", "lt", "python", "-c", KILL)
    while int(sh("docker", "exec", "lt", "python", "-c", COUNT) or 0) < 2 and time.time() - t0 < 60:
        time.sleep(0.2)
    respawn = time.time() - t0
    time.sleep(3)
    stop.set()
    th.join()
    res["worker_crash"].append({"probes": len(log), "failed_probes": log.count(False), "respawn_s": round(respawn, 1)})
    print("A trial", trial + 1, res["worker_crash"][-1], flush=True)
for trial in range(3):
    t0 = time.time()
    sh("docker", "restart", "lt")
    res["container_restart_s"].append(round(wait_healthy() + (time.time() - t0) - (time.time() - t0), 1))
    print("B trial", trial + 1, "restart -> healthy: %.1f s" % (time.time() - t0), flush=True)
    res["container_restart_s"][-1] = round(time.time() - t0, 1)
    time.sleep(3)
MASTER_KILL = ("import os,signal;ps=[int(p) for p in os.listdir('/proc') if p.isdigit() and "
               "b'/usr/local/bin/uvicorn' in open(f'/proc/{p}/cmdline','rb').read()];os.kill(ps[0],signal.SIGKILL)")
res["master_crash_restart_s"] = []
for trial in range(3):
    t0 = time.time()
    sh("docker", "exec", "lt", "python", "-c", MASTER_KILL)
    time.sleep(0.5)
    wait_healthy()
    res["master_crash_restart_s"].append(round(time.time() - t0, 1))
    print("C trial", trial + 1, "master SIGKILL -> healthy again: %.1f s" % res["master_crash_restart_s"][-1], flush=True)
    time.sleep(15)       # restart policy applies back-off after quick failures
print("restart count:", sh("docker", "inspect", "lt", "--format", "{{.RestartCount}}"))
sh("docker", "rm", "-f", "lt")
(ROOT / "artifacts" / "load_test" / "recovery.json").write_text(json.dumps(res, indent=2))
