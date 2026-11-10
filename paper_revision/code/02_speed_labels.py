"""Objective performance labels from measured horizontal running velocity.

The original pipeline labelled athletes with K-Means on the same features the
classifier later used (circular). Here the label comes from a quantity that is
NOT a model input: the athlete's horizontal hip velocity while crossing the
static camera's field of view.

Scale: pixel velocity is divided by a posture-invariant body-size measure,
L = trunk + thigh + shank segment lengths (median over frames), which is
~0.779 x stature (Drillis & Contini segment ratios). Velocity in statures/s is
camera-distance invariant; an approximate m/s value assumes 1.70 m stature.

Tiers: cohort tertiles of velocity (3 recordings per tier).
Output: results/speed_labels.csv, figures/speed_labels.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from common import CLASS_NAMES, CLIPS, FIG, OUT, POSE_DIR, SUBJECT, runner_track

SEG_TO_STATURE = 0.779
ASSUMED_STATURE_M = 1.70
MIN_VIS = 0.5


def seg(df, a, b):
    return np.hypot(df[f"x{a}"] - df[f"x{b}"], df[f"y{a}"] - df[f"y{b}"])


def measure(clip):
    df = pd.read_csv(POSE_DIR / f"{clip}.csv")
    d = runner_track(df)
    d = d[d[["v23", "v24", "v11", "v12"]].min(axis=1) > MIN_VIS].copy()
    hip_x = (d.x23 + d.x24) / 2
    # body size: average of left/right kinematic chains
    L = ((seg(d, 11, 23) + seg(d, 23, 25) + seg(d, 25, 27)) +
         (seg(d, 12, 24) + seg(d, 24, 26) + seg(d, 26, 28))) / 2
    L_med = float(np.median(L))
    slope, intercept, lo, hi = theilslopes(hip_x.values, d.t.values)
    pred = intercept + slope * d.t.values
    r2 = 1 - np.sum((hip_x - pred) ** 2) / np.sum((hip_x - hip_x.mean()) ** 2)
    stature_px = L_med / SEG_TO_STATURE
    v_rel = abs(slope) / stature_px
    return {
        "clip": clip, "subject": SUBJECT[clip], "frames_used": len(d),
        "frames_total": len(df), "span_s": float(d.t.max() - d.t.min()),
        "px_per_s": abs(slope), "stature_px": stature_px, "fit_r2": r2,
        "v_statures_per_s": v_rel,
        "v_statures_per_s_ci": (abs(hi) - abs(lo)) / 2 / stature_px if np.sign(lo) == np.sign(hi) else np.nan,
        "v_approx_mps": v_rel * ASSUMED_STATURE_M,
    }, d.t.values, hip_x.values / stature_px


if __name__ == "__main__":
    rows, traces = [], {}
    for c in CLIPS:
        r, t, x = measure(c)
        rows.append(r)
        traces[c] = (t, x)
    lab = pd.DataFrame(rows).sort_values("v_statures_per_s").reset_index(drop=True)
    q1, q2 = np.quantile(lab.v_statures_per_s, [1 / 3, 2 / 3])
    lab["label"] = np.digitize(lab.v_statures_per_s, [q1, q2])  # 0,1,2
    lab["tier"] = lab.label.map(dict(enumerate(CLASS_NAMES)))
    lab.attrs["thresholds"] = (q1, q2)
    lab.to_csv(OUT / "speed_labels.csv", index=False)
    print(lab[["clip", "subject", "frames_used", "fit_r2", "v_statures_per_s", "v_approx_mps", "tier"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"tertile thresholds (statures/s): {q1:.3f}, {q2:.3f} "
          f"(~{q1*ASSUMED_STATURE_M:.2f}, {q2*ASSUMED_STATURE_M:.2f} m/s)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {0: "#d95f02", 1: "#7570b3", 2: "#1b9e77"}
    for _, r in lab.iterrows():
        t, x = traces[r["clip"]]
        axes[0].plot(t - t.min(), x - x[0], ".", ms=3, color=colors[r.label], label=r["clip"])
    axes[0].set(xlabel="time (s)", ylabel="hip displacement (statures)",
                title="Hip trajectory per recording")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].barh(lab["clip"], lab.v_approx_mps, color=[colors[l] for l in lab.label])
    for q in (q1, q2):
        axes[1].axvline(q * ASSUMED_STATURE_M, ls="--", color="grey")
    axes[1].set(xlabel="velocity (m/s, assuming 1.70 m stature)", title="Velocity-based tiers")
    fig.tight_layout()
    fig.savefig(FIG / "speed_labels.png", dpi=200)
