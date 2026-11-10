"""Measured inference latency (reviewer comment 6).

Times each pipeline stage on the machine it runs on and writes a JSON record
with the hardware/software description, so every team member can run it on
their own device and the paper reports only measured numbers.

    python 06_latency.py                      # all cores
    taskpolicy -b python 06_latency.py        # macOS: efficiency cores only

Stages: video decode, MediaPipe Pose per frame (complexity 0/1/2), feature
extraction, classifier inference (Keras and TFLite, batch of 1).
Output: results/latency_<host>_<tag>.json, results/sprint_mlp.tflite
"""
import json
import os
import platform
import socket
import subprocess
import sys
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd

from common import CLIPS, OUT, video_path

TAG = sys.argv[1] if len(sys.argv) > 1 else "default"
N_REPEAT_MODEL = 2000


def cpu_name():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def stats(ms):
    ms = np.asarray(ms)
    return {"mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "n": int(len(ms))}


def bench_pose(complexity, clips):
    from mediapipe.python.solutions import pose as mp_pose
    dec, pose_ms, per_clip = [], [], []
    for c in clips:
        cap = cv2.VideoCapture(str(video_path(c)))
        t_clip = time.perf_counter()
        with mp_pose.Pose(model_complexity=complexity, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            while True:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                t1 = time.perf_counter()
                if not ok:
                    break
                pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                t2 = time.perf_counter()
                dec.append((t1 - t0) * 1e3)
                pose_ms.append((t2 - t1) * 1e3)
        per_clip.append((time.perf_counter() - t_clip) * 1e3)
        cap.release()
    return {"decode_per_frame": stats(dec), "pose_per_frame": stats(pose_ms),
            "video_clip_total": stats(per_clip)}


def bench_features():
    import importlib
    f = importlib.import_module("03_features")
    v, i = [], []
    for c in CLIPS:
        t0 = time.perf_counter(); _, span = f.video_features(c); t1 = time.perf_counter()
        f.imu_features(c, span); t2 = time.perf_counter()
        v.append((t1 - t0) * 1e3); i.append((t2 - t1) * 1e3)
    return {"video_features": stats(v), "imu_features": stats(i)}


def bench_model():
    import tensorflow as tf
    sys.path.insert(0, os.path.dirname(__file__))
    import importlib
    ev = importlib.import_module("05_evaluate")
    tf.keras.utils.set_random_seed(42)
    m = ev.make_mlp(19)
    x = np.random.default_rng(0).normal(size=(1, 19)).astype(np.float32)
    m.predict(x, verbose=0)
    k = []
    for _ in range(200):
        t0 = time.perf_counter(); m.predict(x, verbose=0); k.append((time.perf_counter() - t0) * 1e3)
    kc = []
    for _ in range(N_REPEAT_MODEL):
        t0 = time.perf_counter(); m(x, training=False); kc.append((time.perf_counter() - t0) * 1e3)
    tfl_path = OUT / "sprint_mlp.tflite"
    tfl_path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(m).convert())
    it = tf.lite.Interpreter(model_path=str(tfl_path), num_threads=1)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    t = []
    for _ in range(N_REPEAT_MODEL):
        t0 = time.perf_counter(); it.set_tensor(inp, x); it.invoke(); it.get_tensor(out)
        t.append((time.perf_counter() - t0) * 1e3)
    return {"keras_predict": stats(k), "keras_call": stats(kc), "tflite_1thread": stats(t),
            "tflite_size_kb": tfl_path.stat().st_size / 1024}


if __name__ == "__main__":
    import mediapipe, tensorflow, sklearn
    rec = {
        "host": socket.gethostname(), "tag": TAG, "cpu": cpu_name(), "os": platform.platform(),
        "python": platform.python_version(), "cores_logical": os.cpu_count(),
        "mediapipe": mediapipe.__version__, "tensorflow": tensorflow.__version__,
        "opencv": cv2.__version__, "sklearn": sklearn.__version__,
        "video": "1920x1080 HEVC 60 fps, 9 clips",
    }
    rec["model"] = bench_model()
    rec["features"] = bench_features()
    rec["pose"] = {f"complexity_{c}": bench_pose(c, CLIPS) for c in (0, 1, 2)}
    out = OUT / f"latency_{rec['host'].split('.')[0]}_{TAG}.json"
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ("cpu", "tag")}, indent=None))
    for c, r in rec["pose"].items():
        print(f"{c}: pose/frame median {r['pose_per_frame']['median_ms']:.1f} ms "
              f"p95 {r['pose_per_frame']['p95_ms']:.1f} | decode {r['decode_per_frame']['median_ms']:.1f} ms "
              f"| clip total median {r['video_clip_total']['median_ms']:.0f} ms")
    m = rec["model"]
    print(f"MLP keras.predict {m['keras_predict']['median_ms']:.2f} ms, direct call "
          f"{m['keras_call']['median_ms']:.3f} ms, TFLite {m['tflite_1thread']['median_ms']:.4f} ms "
          f"({m['tflite_size_kb']:.0f} KB)")
    f = rec["features"]
    print(f"features: video {f['video_features']['median_ms']:.1f} ms, imu {f['imu_features']['median_ms']:.1f} ms")
    print("saved", out)
