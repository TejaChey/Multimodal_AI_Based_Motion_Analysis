"""Objective performance labels from measured running velocity.

Velocity is the horizontal hip speed while crossing the static camera's field of view,
normalised by a posture-invariant body length (trunk + thigh + shank = 0.779 x stature),
so it is independent of camera distance. Tiers are cohort tertiles.
"""
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .pose import runner_track


def _seg(df, a, b):
    return np.hypot(df[f"x{a}"] - df[f"x{b}"], df[f"y{a}"] - df[f"y{b}"])


def measure_velocity(pose: pd.DataFrame, p: dict) -> dict:
    cfg = p["labels"]
    d = runner_track(pose)
    d = d[d[["v23", "v24", "v11", "v12"]].min(axis=1) > cfg["min_visibility"]]
    if len(d) < 10:
        raise ValueError("runner not tracked: fewer than 10 usable frames")
    hip_x = ((d.x23 + d.x24) / 2).values
    L = (((_seg(d, 11, 23) + _seg(d, 23, 25) + _seg(d, 25, 27)) +
          (_seg(d, 12, 24) + _seg(d, 24, 26) + _seg(d, 26, 28))) / 2).median()
    stature_px = float(L) / cfg["seg_to_stature"]
    slope, icpt, _, _ = theilslopes(hip_x, d.t.values)
    resid = hip_x - (icpt + slope * d.t.values)
    r2 = 1 - np.sum(resid ** 2) / max(np.sum((hip_x - hip_x.mean()) ** 2), 1e-12)
    v = abs(slope) / stature_px
    return {"v_statures_per_s": float(v), "v_mps": float(v * cfg["assumed_stature_m"]),
            "fit_r2": float(r2), "frames_used": int(len(d)), "stature_px": stature_px,
            "span_s": float(d.t.max() - d.t.min())}


def tier_thresholds(v) -> tuple:
    q = np.quantile(np.asarray(v, float), [1 / 3, 2 / 3])
    return float(q[0]), float(q[1])


def assign_tiers(v, thresholds) -> np.ndarray:
    """0 = Beginner, 1 = Intermediate, 2 = Elite."""
    return np.digitize(np.asarray(v, float), list(thresholds))
