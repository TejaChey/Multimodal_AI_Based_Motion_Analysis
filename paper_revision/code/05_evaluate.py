"""Leakage-free, subject-independent evaluation (reviewer comments 2 and 4).

Protocol
  * Leave-One-Subject-Out (LOSO): 8 folds; both Yeshwanth trials (S8) are always
    held out together, so no athlete is ever in train and test at once.
  * Inside each fold, on TRAINING subjects only: StandardScaler fit, then
    augmentation (Gaussian jitter at 5% of feature SD + same-class linear
    interpolation) to 300 samples/class. The held-out athlete never seeds
    synthetic data.
  * Labels: velocity tertiles (02_speed_labels.py); features: 03_features.py.
  * Metrics on the 9 pooled out-of-fold predictions: accuracy, balanced
    accuracy, macro precision/recall/F1, confusion matrix, 95% bootstrap CI
    (resampling subjects), permutation-test p-value (labels shuffled).
  * Sensitivity analysis: IMU and fused models re-run without the 2 recordings
    whose watch file does not match the video (04_sync_validation.py).
  * Regression check: LOSO ridge regression of velocity (continuous) from
    features, which uses the information in the data more efficiently than 3 tiers.

Output: results/eval_*.csv, figures/confusion_*.png
"""
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             mean_absolute_error, precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CLASS_NAMES, FIG, OUT, SEED

VIDEO = ["knee_angle_mean", "hip_angle_mean", "knee_sym_index", "hip_sym_index", "trunk_lean_deg",
         "vert_osc_stature", "step_freq_video_hz", "gct_ms", "duty_factor"]
IMU = ["peak_accel", "mean_accel", "std_accel", "rms_jerk", "peak_gyro", "mean_gyro", "std_gyro",
       "arm_freq_gyro_hz", "sparc_accel", "sparc_gyro"]
SETS = {"Video-only": VIDEO, "IMU-only": IMU, "Fused": VIDEO + IMU}
IMU_MISMATCH = ["Chandra1", "Jeeva"]
N_PER_CLASS = 300
N_PERM = int(os.environ.get("N_PERM", 200))
N_BOOT = 2000


def augment(X, y, rng, n=N_PER_CLASS, noise=0.05):
    """Training-fold-only augmentation in standardised space."""
    Xs, ys = [X], [y]
    for c in np.unique(y):
        Xc = X[y == c]
        k = n - len(Xc)
        base = Xc[rng.integers(0, len(Xc), k)]
        if len(Xc) >= 2:  # half interpolation, half jitter
            other = Xc[rng.integers(0, len(Xc), k)]
            a = rng.uniform(0.1, 0.9, (k, 1))
            interp = rng.random(k) < 0.5
            base = np.where(interp[:, None], base + a * (other - base), base)
        Xs.append(base + rng.normal(0, noise, base.shape))
        ys.append(np.full(k, c))
    return np.vstack(Xs), np.concatenate(ys)


def make_mlp(dim):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
    from tensorflow.keras.regularizers import l2
    m = Sequential([Input((dim,))] + [
        layer for units, p in ((128, .4), (64, .3), (32, .2))
        for layer in (Dense(units, activation="relu", kernel_regularizer=l2(1e-3)),
                      BatchNormalization(), Dropout(p))] + [Dense(3, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


class KerasMLP:
    """Same architecture as the paper / deployed model (Phase_3)."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed)
        self.m = make_mlp(X.shape[1])
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
        self.m.fit(X, y, epochs=100, batch_size=32, validation_split=0.15, callbacks=cb, verbose=0)
        return self

    def predict(self, X):
        return self.m.predict(X, verbose=0).argmax(1)


MODELS = {
    "LogReg": lambda s: LogisticRegression(C=1.0, max_iter=2000),
    "SVM-RBF": lambda s: SVC(C=1.0, kernel="rbf", random_state=s),
    "RF": lambda s: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=s),
    "MLP": lambda s: KerasMLP(s),
}


def loso_predict(df, feats, model, augment_on, seed=SEED, y=None):
    y = df.label.values if y is None else y
    X = df[feats].values
    pred = np.empty(len(df), int)
    rng = np.random.default_rng(seed)
    for subj in df.subject.unique():
        te = (df.subject == subj).values
        sc = StandardScaler().fit(X[~te])
        Xtr, ytr = sc.transform(X[~te]), y[~te]
        if augment_on:
            Xtr, ytr = augment(Xtr, ytr, rng)
        pred[te] = MODELS[model](seed).fit(Xtr, ytr).predict(sc.transform(X[te]))
    return pred


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], average="macro", zero_division=0)
    return {"acc": accuracy_score(y, p), "bal_acc": balanced_accuracy_score(y, p),
            "precision": pr, "recall": rc, "f1": f1}


def bootstrap_ci(df, y, p, rng):
    subj = df.subject.values
    us = np.unique(subj)
    accs, f1s = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(us, len(us))
        idx = np.concatenate([np.where(subj == s)[0] for s in pick])
        accs.append(accuracy_score(y[idx], p[idx]))
        f1s.append(precision_recall_fscore_support(y[idx], p[idx], labels=[0, 1, 2], average="macro",
                                                   zero_division=0)[2])
    return np.percentile(accs, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5])


def perm_p(df, feats, model, aug, observed, rng, n):
    y = df.label.values
    hits = 0
    for _ in range(n):
        yp = rng.permutation(y)
        hits += accuracy_score(yp, loso_predict(df, feats, model, aug, y=yp)) >= observed
    return (hits + 1) / (n + 1)


def run(df, tag, models, aug_modes, n_perm):
    rng = np.random.default_rng(SEED)
    rows, preds = [], {}
    y = df.label.values
    for mset, feats in SETS.items():
        for model in models:
            for aug in aug_modes:
                seeds = [SEED + i for i in range(5)] if model == "MLP" else [SEED]
                ps = [loso_predict(df, feats, model, aug, seed=s) for s in seeds]
                ms = pd.DataFrame([metrics(y, p) for p in ps])
                p = ps[0]
                (alo, ahi), (flo, fhi) = bootstrap_ci(df, y, p, rng)
                n = n_perm if model != "MLP" else 0
                pv = perm_p(df, feats, model, aug, ms.acc.iloc[0], rng, n) if n else np.nan
                row = {"subset": tag, "modality": mset, "model": model, "augmentation": aug,
                       **ms.mean().to_dict(), "acc_sd_over_seeds": ms.acc.std() if len(ms) > 1 else 0.0,
                       "acc_ci_lo": alo, "acc_ci_hi": ahi, "f1_ci_lo": flo, "f1_ci_hi": fhi,
                       "perm_p": pv, "n": len(df)}
                rows.append(row)
                preds[(mset, model, aug)] = p
                print(f"[{tag}] {mset:10s} {model:7s} aug={aug!s:5s} acc={row['acc']:.3f} "
                      f"(sd {row['acc_sd_over_seeds']:.3f}) F1={row['f1']:.3f} "
                      f"CI=[{alo:.2f},{ahi:.2f}] p={pv:.3f}", flush=True)
    return pd.DataFrame(rows), preds


def plot_cm(y, p, title, path):
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(3, cm.max()))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 1.5 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted tier", ylabel="True tier (velocity)", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def regression(df, tag):
    out = []
    for mset, feats in SETS.items():
        X, v = df[feats].values, df.v_statures_per_s.values * 1.70
        pred = np.empty(len(df))
        for subj in df.subject.unique():
            te = (df.subject == subj).values
            sc = StandardScaler().fit(X[~te])
            m = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(sc.transform(X[~te]), v[~te])
            pred[te] = m.predict(sc.transform(X[te]))
        base = np.array([v[(df.subject != s).values].mean() for s in df.subject])
        out.append({"subset": tag, "modality": mset, "MAE_mps": mean_absolute_error(v, pred),
                    "MAE_baseline_mean_mps": mean_absolute_error(v, base),
                    "pearson_r": np.corrcoef(v, pred)[0, 1]})
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = pd.read_csv(OUT / "features.csv")
    res_all, preds = run(df, "all_9", list(MODELS), [False, True], N_PERM)
    sub = df[~df["clip"].isin(IMU_MISMATCH)].reset_index(drop=True)
    res_sub, _ = run(sub, "imu_matched_7", ["LogReg", "SVM-RBF", "RF"], [False, True], N_PERM)
    res = pd.concat([res_all, res_sub])
    res.to_csv(OUT / "eval_classification.csv", index=False)
    reg = pd.concat([regression(df, "all_9"), regression(sub, "imu_matched_7")])
    reg.to_csv(OUT / "eval_regression.csv", index=False)
    print(reg.round(3).to_string(index=False))
    y = df.label.values
    for key in [("Fused", "MLP", True), ("Video-only", "MLP", True), ("IMU-only", "MLP", True)]:
        plot_cm(y, preds[key], f"{key[0]} {key[1]} (LOSO)", FIG / f"confusion_{key[0]}_{key[1]}.png")
    # also the best-performing classical model per modality
    best = res_all[res_all.model != "MLP"].sort_values("acc", ascending=False).groupby("modality").head(1)
    for _, r in best.iterrows():
        k = (r.modality, r.model, r.augmentation)
        plot_cm(y, preds[k], f"{k[0]} {k[1]} (LOSO)", FIG / f"confusion_{k[0]}_{k[1]}_best.png")
