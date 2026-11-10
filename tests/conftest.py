"""Shared fixtures: a synthetic sprinter (known speed) and a synthetic feature table."""
import numpy as np
import pandas as pd
import pytest

from sprintml.config import load_params
from sprintml.features import FEATURES


@pytest.fixture(scope="session")
def params():
    """Real parameters, with a lighter candidate set / augmentation so the suite runs in seconds."""
    p = load_params()
    p["train"] = dict(p["train"], n_aug_per_class=60,
                      candidates=[{"features": f, "model": "logreg", "augment": True} for f in ("video", "imu", "fused")])
    return p


def synthetic_pose(speed_statures_per_s=4.5, stature_px=200.0, fps=60.0, seconds=1.4, false_frames=30,
                   cadence_hz=4.0, seed=0):
    """Side-view runner moving right-to-left at a known speed, preceded by a static false detection."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    seg = stature_px * 0.779          # trunk+thigh+shank
    trunk, thigh, shank = seg * 0.34, seg * 0.33, seg * 0.33
    rows = []
    # static false detection (a "person" in the stands) before the runner enters
    for i in range(false_frames):
        r = {"frame": i, "t": i / fps, "detected": True, "fps": fps, "W": 1920, "H": 1080, "pose_ms": 5.0}
        for k in range(33):
            r.update({f"x{k}": 300.0, f"y{k}": 150.0, f"z{k}": 0.0, f"v{k}": 0.9})
        r.update({"x11": 300, "y11": 120, "x12": 300, "y12": 120, "x23": 300, "y23": 150, "x24": 300, "y24": 150,
                  "x25": 300, "y25": 170, "x26": 300, "y26": 170, "x27": 300, "y27": 190, "x28": 300, "y28": 190})
        rows.append(r)
    v_px = speed_statures_per_s * stature_px
    for j in range(n):
        i = false_frames + j
        t = j / fps
        hx = 1600 - v_px * t + rng.normal(0, 0.3)
        hy = 500 + 4 * np.sin(2 * np.pi * cadence_hz * 2 * t)
        ph = 2 * np.pi * cadence_hz * t
        r = {"frame": i, "t": i / fps, "detected": True, "fps": fps, "W": 1920, "H": 1080, "pose_ms": 8.0}
        for k in range(33):
            r.update({f"x{k}": hx, f"y{k}": hy, f"z{k}": 0.0, f"v{k}": 0.95})
        sh_y = hy - trunk
        for side, off in ((11, 0), (12, np.pi)):
            r.update({f"x{side}": hx, f"y{side}": sh_y})
        for hip, kn, an, off in ((23, 25, 27, 0), (24, 26, 28, np.pi)):
            a = 0.6 * np.sin(ph + off)          # thigh swing angle
            b = a - 0.5                         # shank angle (knee flexion)
            kx, ky = hx - thigh * np.sin(a), hy + thigh * np.cos(a)
            r.update({f"x{hip}": hx, f"y{hip}": hy, f"x{kn}": kx, f"y{kn}": ky,
                      f"x{an}": kx - shank * np.sin(b), f"y{an}": ky + shank * np.cos(b)})
        for w, off in ((15, 0), (16, np.pi)):
            r.update({f"x{w}": hx - 0.35 * trunk * np.sin(ph + off + np.pi), f"y{w}": sh_y + trunk * 0.4})
        rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture
def pose_df():
    return synthetic_pose()


def synthetic_imu(seconds=1.6, hz=100.0, cadence=2.0, seed=0, t0_ns=1_774_000_000_000_000_000):
    rng = np.random.default_rng(seed)
    n = int(seconds * hz)
    t = np.arange(n) / hz
    base = t0_ns + (t * 1e9).astype("int64")
    a = pd.DataFrame({"time": base, "seconds_elapsed": t,
                      "x": 20 * np.sin(2 * np.pi * cadence * t) + rng.normal(0, 1, n),
                      "y": 30 + 15 * np.sin(2 * np.pi * 2 * cadence * t) + rng.normal(0, 1, n),
                      "z": 9 + rng.normal(0, 1, n)})
    g = pd.DataFrame({"time": base, "seconds_elapsed": t, "x": 2 * np.sin(2 * np.pi * cadence * t),
                      "y": 8 * np.sin(2 * np.pi * cadence * t + 1) + rng.normal(0, .2, n),
                      "z": rng.normal(0, .2, n)})
    return a, g


def synthetic_table(n_per_class=3, seed=0, separable=True):
    """Feature table with a velocity-correlated signal (separable) or pure noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(3):
        for i in range(n_per_class):
            f = rng.normal(0, 1, len(FEATURES))
            if separable:
                f += 2.5 * c          # class signal on every feature
            rows.append({"clip": f"c{c}{i}", "subject": f"S{c}{i}",
                         "v_statures_per_s": 4.0 + 0.5 * c + rng.normal(0, 0.03), **dict(zip(FEATURES, f))})
    df = pd.DataFrame(rows)
    return df
