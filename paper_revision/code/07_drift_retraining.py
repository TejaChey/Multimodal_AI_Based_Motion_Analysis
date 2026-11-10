"""KS-test drift thresholds and retraining validation (reviewer comments 7, 8).

A. Threshold justification (Monte Carlo): per-feature two-sample KS test,
   Bonferroni alpha = 0.05 / n_features so the family-wise false-alarm rate is
   <= 5%. Power vs. monitoring-window size m and reference size n_ref for a
   shift of delta SD in a single feature -> minimum window size.
B. Device-shift scenarios built from the REAL signals (features re-extracted):
   camera 30 fps (every 2nd pose frame), watch 50 Hz (every 2nd IMU sample),
   watch gain +15%, and camera 30 fps + watch 50 Hz. Report KS drift flags and
   LOSO accuracy of a frozen model vs. a model retrained on shifted data.
   Labels come from measured velocity, so retraining needs no manual annotation.
C. Changing athlete population (real chronology): recordings ordered by
   capture time. All Beginner-tier athletes were recorded last, so a model
   trained on the early athletes never saw that tier. Compare a frozen model
   with retraining after each newly labelled recording (prequential).

Output: results/drift_*.csv, figures/drift_power.png
"""
import importlib
import os
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from common import CLIPS, FIG, OUT, POSE_DIR, SEED, SUBJECT

F = importlib.import_module("03_features")
E = importlib.import_module("05_evaluate")
FEATS = E.VIDEO + E.IMU
ALPHA = 0.05 / len(FEATS)
MODEL = os.environ.get("DRIFT_MODEL", "LogReg")
CAPTURE_ORDER = ["Hanshi2", "Chandra1", "Jeeva", "Amith", "Mohammed_araar", "Sachin", "Tejas",
                 "Yeshwanth", "Yeshwanth_trail_2"]  # from video creation_time


# ── A. power analysis ─────────────────────────────────────────────────────────
def power(n_ref, m, delta, n_feat=len(FEATS), reps=2000, rng=np.random.default_rng(SEED)):
    hits = 0
    for _ in range(reps):
        ref = rng.normal(size=(n_ref, n_feat))
        new = rng.normal(size=(m, n_feat))
        new[:, 0] += delta
        hits += any(ks_2samp(ref[:, j], new[:, j]).pvalue < ALPHA for j in range(n_feat))
    return hits / reps


def part_a():
    rows = []
    for n_ref in (9, 30, 100, 900):
        for m in (5, 10, 20, 30, 50, 100):
            for delta in (0.0, 0.5, 1.0, 2.0):
                reps = 1000 if n_ref < 900 else 400
                rows.append({"n_ref": n_ref, "window_m": m, "delta_sd": delta,
                             "detect_rate": power(n_ref, m, delta, reps=reps)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "drift_power.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, n_ref in zip(axes, (9, 30, 100, 900)):
        for delta, c in zip((0.0, 0.5, 1.0, 2.0), ("grey", "#7570b3", "#d95f02", "#1b9e77")):
            s = df[(df.n_ref == n_ref) & (df.delta_sd == delta)]
            ax.plot(s.window_m, s.detect_rate, "o-", color=c, label=f"shift {delta} SD")
        ax.axhline(0.8, ls=":", color="k"); ax.axhline(0.05, ls=":", color="grey")
        ax.set(title=f"reference n = {n_ref}", xlabel="monitoring window m", xscale="log")
    axes[0].set_ylabel("P(drift alarm)"); axes[0].legend(fontsize=7)
    fig.suptitle(f"Per-feature KS, Bonferroni α = 0.05/{len(FEATS)} = {ALPHA:.4f}", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "drift_power.png", dpi=200)
    return df


# ── B. device-shift scenarios on real signals ─────────────────────────────────
def features_under(pose_step=1, imu_step=1, acc_gain=1.0):
    tmp = Path(tempfile.mkdtemp())
    orig_pose, orig_load = F.POSE_DIR, F.load_imu
    for c in CLIPS:
        p = pd.read_csv(POSE_DIR / f"{c}.csv")
        p = p[p.frame % pose_step == 0].copy()
        p["frame"] //= pose_step
        p["fps"] = p.fps / pose_step
        p.to_csv(tmp / f"{c}.csv", index=False)

    def load_shifted(clip):
        m = orig_load(clip).iloc[::imu_step].reset_index(drop=True)
        for a in ("x_a", "y_a", "z_a"):
            m[a] *= acc_gain
        return m

    F.POSE_DIR, F.load_imu = tmp, load_shifted
    try:
        rows = []
        for c in CLIPS:
            v, span = F.video_features(c)
            i, _ = F.imu_features(c, span)
            rows.append({"clip": c, **v, **i})
    finally:
        F.POSE_DIR, F.load_imu = orig_pose, orig_load
    return pd.DataFrame(rows).set_index("clip").loc[CLIPS]


def loso(train_df, test_df, y, groups):
    pred = np.empty(len(y), int)
    for s in np.unique(groups):
        te = groups == s
        med = np.nanmedian(train_df[~te], axis=0)  # impute features a degraded device cannot measure
        tr = np.where(np.isnan(train_df[~te]), med, train_df[~te])
        ts = np.where(np.isnan(test_df[te]), med, test_df[te])
        sc = StandardScaler().fit(tr)
        pred[te] = E.MODELS[MODEL](SEED).fit(sc.transform(tr), y[~te]).predict(sc.transform(ts))
    return accuracy_score(y, pred)


def part_b():
    base = pd.read_csv(OUT / "features.csv").set_index("clip").loc[CLIPS]
    y, groups = base.label.values, np.array([SUBJECT[c] for c in CLIPS])
    X0 = base[FEATS].values
    scen = {"camera 30 fps": dict(pose_step=2), "watch 50 Hz": dict(imu_step=2),
            "watch gain +15%": dict(acc_gain=1.15),
            "camera 30 fps + watch 50 Hz": dict(pose_step=2, imu_step=2)}
    rows, flags = [], []
    acc_base = loso(X0, X0, y, groups)
    for name, kw in scen.items():
        Xs = features_under(**kw)[FEATS].values
        p = np.array([ks_2samp(X0[:, j], Xs[~np.isnan(Xs[:, j]), j]).pvalue for j in range(len(FEATS))])
        shifted = [FEATS[j] for j in np.argsort(p)[:3]]
        rel = np.nanmedian(np.abs(Xs - X0) / (X0.std(0) + 1e-9), axis=0)
        rows.append({
            "scenario": name, "model": MODEL,
            "unmeasurable_values": ", ".join(f"{CLIPS[i]}:{FEATS[j]}" for i, j in zip(*np.where(np.isnan(Xs)))),
            "ks_alarm_bonferroni": bool((p < ALPHA).any()),
            "n_features_p_lt_0.05": int((p < 0.05).sum()), "min_ks_p": p.min(),
            "most_shifted_features": ", ".join(shifted),
            "max_median_shift_sd": rel.max(),
            "acc_no_shift": acc_base,
            "acc_frozen_model": loso(X0, Xs, y, groups),
            "acc_retrained": loso(Xs, Xs, y, groups),
        })
        flags.append(pd.Series(rel, index=FEATS, name=name))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "drift_device_scenarios.csv", index=False)
    pd.concat(flags, axis=1).to_csv(OUT / "drift_feature_shift_sd.csv")
    return out


# ── C. chronological population shift ────────────────────────────────────────
def part_c(n_init=4):
    df = pd.read_csv(OUT / "features.csv").set_index("clip").loc[CAPTURE_ORDER]
    X, y = df[FEATS].values, df.label.values
    rows = []
    for policy in ("frozen", "retrain_each"):
        tr = list(range(n_init))
        for k in range(n_init, len(df)):
            sc = StandardScaler().fit(X[tr])
            m = E.MODELS[MODEL](SEED).fit(sc.transform(X[tr]), y[tr])
            pred = int(m.predict(sc.transform(X[[k]]))[0])
            ks_p = min(ks_2samp(X[:n_init, j], X[n_init:k + 1, j]).pvalue for j in range(len(FEATS)))
            rows.append({"policy": policy, "step": k - n_init + 1, "clip": CAPTURE_ORDER[k],
                         "true": E.CLASS_NAMES[y[k]], "pred": E.CLASS_NAMES[pred],
                         "correct": pred == y[k], "classes_seen": sorted(set(y[tr])),
                         "stream_min_ks_p": ks_p, "ks_alarm": ks_p < ALPHA})
            if policy == "retrain_each":
                tr.append(k)  # label arrives automatically from measured velocity
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "drift_chronological.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    a = pd.read_csv(OUT / "drift_power.csv") if os.environ.get("SKIP_A") else part_a()
    print("A. detection rate (n_ref=9 and 100):")
    print(a[a.n_ref.isin([9, 100])].pivot_table(index=["n_ref", "window_m"], columns="delta_sd",
                                                values="detect_rate").round(2).to_string())
    print("\nB. device-shift scenarios"); print(part_b().round(3).to_string(index=False))
    c = part_c()
    print("\nC. chronological stream"); print(c.round(4).to_string(index=False))
    print(c.groupby("policy").correct.mean())
