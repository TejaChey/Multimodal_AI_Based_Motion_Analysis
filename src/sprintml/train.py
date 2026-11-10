"""Subject-independent training, MLflow tracking and the model promotion gate.

Protocol (no leakage): leave-one-subject-out; scaling and augmentation are fitted on the
training athletes of each fold only. The candidate (feature set x model) is chosen by LOSO
macro-F1, so the permutation p-value uses the MAX statistic over all candidates (it accounts
for the selection). A challenger is promoted only if it clears the gate; the very first model
is registered as a *provisional* champion so the service has something to serve, and is
flagged as not validated whenever p >= alpha.
"""
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from .config import CLASS_NAMES, MODEL_NAME, art_dir, dvc_data_hash, git_commit
from .features import FEATURES, IMU, VIDEO
from .labels import assign_tiers, tier_thresholds

SETS = {"video": VIDEO, "imu": IMU, "fused": FEATURES}


# ── model pieces ─────────────────────────────────────────────────────────────
def make_clf(name, seed):
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
    raise ValueError(name)


def augment(X, y, rng, n, noise):
    """Training-fold-only augmentation in standardised space: jitter + same-class interpolation."""
    Xs, ys = [X], [y]
    for c in np.unique(y):
        Xc = X[y == c]
        k = n - len(Xc)
        if k <= 0:
            continue
        base = Xc[rng.integers(0, len(Xc), k)]
        if len(Xc) >= 2:
            other = Xc[rng.integers(0, len(Xc), k)]
            a = rng.uniform(0.1, 0.9, (k, 1))
            base = np.where((rng.random(k) < 0.5)[:, None], base + a * (other - base), base)
        Xs.append(base + rng.normal(0, noise, base.shape))
        ys.append(np.full(k, c))
    return np.vstack(Xs), np.concatenate(ys)


def loso_predict(X, y, subjects, cand, p, seed, y_fit=None):
    y_fit = y if y_fit is None else y_fit
    pred = np.empty(len(y), int)
    rng = np.random.default_rng(seed)
    for s in np.unique(subjects):
        te = subjects == s
        sc = StandardScaler().fit(X[~te])
        Xtr, ytr = sc.transform(X[~te]), y_fit[~te]
        if cand["augment"]:
            Xtr, ytr = augment(Xtr, ytr, rng, p["train"]["n_aug_per_class"], p["train"]["aug_noise"])
        pred[te] = make_clf(cand["model"], seed).fit(Xtr, ytr).predict(sc.transform(X[te]))
    return pred


def macro_f1(y, pred):
    return f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)


def _null_max(df, cands, y_perm, seed, p):
    """Max macro-F1 over all candidates under one label permutation."""
    subj = df.subject.values
    return max(macro_f1(y_perm, loso_predict(df[SETS[c["features"]]].values, y_perm, subj, c, p, seed))
               for c in cands)


# ── data sufficiency + evaluation ────────────────────────────────────────────
def check_trainable(df, p):
    g = p["gate"]
    counts = np.bincount(df.label.values, minlength=3)
    if (counts == 0).any():
        return False, f"a tier has no recordings (counts {counts.tolist()})"
    if counts.min() < g["min_samples_per_class"]:
        return False, f"fewer than {g['min_samples_per_class']} recordings in a tier (counts {counts.tolist()})"
    if df.subject.nunique() < g["min_subjects"]:
        return False, f"fewer than {g['min_subjects']} athletes ({df.subject.nunique()})"
    return True, "ok"


def evaluate(df, p, n_perm=None, n_jobs=-1):
    """LOSO over all candidates; max-statistic permutation p-value."""
    seed, n_perm = p["train"]["seed"], p["train"]["n_perm"] if n_perm is None else n_perm
    y, subj, cands = df.label.values, df.subject.values, p["train"]["candidates"]
    results = []
    for c in cands:
        pred = loso_predict(df[SETS[c["features"]]].values, y, subj, c, p, seed)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        results.append({**c, "accuracy": accuracy_score(y, pred), "precision": pr, "recall": rc, "f1": f1,
                        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()})
    best = max(results, key=lambda r: r["f1"])
    # permute labels at subject level (a subject's trials keep one label)
    sub_ids = np.unique(subj)
    sub_lab = {s: y[subj == s][0] for s in sub_ids}
    rng = np.random.default_rng(seed + 1)
    perms = []
    for _ in range(n_perm):
        shuffled = rng.permutation([sub_lab[s] for s in sub_ids])
        perms.append(np.array([dict(zip(sub_ids, shuffled))[s] for s in subj]))
    null = Parallel(n_jobs=n_jobs)(delayed(_null_max)(df, cands, yp, seed, p) for yp in perms) if n_perm else []
    pval = (1 + sum(v >= best["f1"] - 1e-12 for v in null)) / (1 + len(null)) if n_perm else float("nan")
    return {"candidates": results, "best": best, "perm_p": float(pval), "n_perm": n_perm,
            "chance_acc": float(np.max(np.bincount(y)) / len(y)), "n": int(len(df)),
            "n_subjects": int(len(sub_ids))}


def fit_final(df, cand, p):
    X, y = df[SETS[cand["features"]]].values, df.label.values
    seed = p["train"]["seed"]
    sc = StandardScaler().fit(X)
    Xs, ys = sc.transform(X), y
    if cand["augment"]:
        Xs, ys = augment(Xs, ys, np.random.default_rng(seed), p["train"]["n_aug_per_class"], p["train"]["aug_noise"])
    clf = make_clf(cand["model"], seed).fit(Xs, ys)
    return {"scaler": sc, "clf": clf, "features": SETS[cand["features"]], "classes": CLASS_NAMES}


# ── promotion gate ───────────────────────────────────────────────────────────
def gate_decision(challenger: dict, champion_meta, p) -> dict:
    """Pure function: returns {promote, validated, reasons}."""
    alpha = p["gate"]["alpha"]
    validated = bool(challenger["perm_p"] == challenger["perm_p"] and challenger["perm_p"] < alpha)
    if champion_meta is None:
        return {"promote": True, "validated": validated, "bootstrap": True,
                "reasons": ["no champion yet: registered as PROVISIONAL" + ("" if validated else
                            " (not statistically above chance)")]}
    reasons = []
    if not validated:
        reasons.append(f"not significantly above chance (permutation p={challenger['perm_p']:.3f} >= {alpha})")
    if challenger["best"]["f1"] < float(champion_meta.get("loso_macro_f1", 0)):
        reasons.append(f"macro-F1 {challenger['best']['f1']:.3f} < champion {float(champion_meta['loso_macro_f1']):.3f}")
    return {"promote": not reasons, "validated": validated, "bootstrap": False, "reasons": reasons}


# ── MLflow registry + export ────────────────────────────────────────────────
def _client():
    import mlflow
    from mlflow import MlflowClient
    d = art_dir()
    mlflow.set_tracking_uri(f"sqlite:///{d / 'mlflow.db'}")
    return mlflow, MlflowClient()


def current_champion():
    mlflow, client = _client()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        return {**mv.tags, "version": str(mv.version)}
    except Exception:
        return None


def export_champion(bundle: dict, meta: dict, directory=None, reference: pd.DataFrame = None) -> Path:
    """Self-contained serving directory: model + metadata + the reference set for drift monitoring."""
    d = Path(directory or art_dir() / "champion")
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "model.joblib")
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    if reference is not None:
        reference.to_csv(d / "reference_features.csv", index=False)
    return d


def restore_champion() -> bool:
    """Re-create artifacts/champion from the registry (e.g. after `dvc repro` cleared it)."""
    import mlflow
    d = art_dir() / "champion"
    if (d / "model.joblib").exists():
        return True
    mlflow_, client = _client()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception:
        return False
    src = Path(mlflow.artifacts.download_artifacts(artifact_uri=mv.source))
    shutil.copytree(src, d, dirs_exist_ok=True)
    meta = json.loads((d / "meta.json").read_text())
    meta["version"] = str(mv.version)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return True


def train_and_register(df, p, reason="manual", n_perm=None) -> dict:
    """Evaluate, log to MLflow, apply the promotion gate, export the champion if promoted."""
    t0 = time.time()
    ok, why = check_trainable(df, p)
    restore_champion()
    mlflow, client = _client()
    exp = p["mlops"]["experiment_name"]
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=(art_dir() / "mlruns").as_uri())
    mlflow.set_experiment(exp)
    champ = current_champion()
    base = {"reason": reason, "n": int(len(df)), "n_subjects": int(df.subject.nunique()),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with mlflow.start_run(run_name=f"train-{reason}") as run:
        mlflow.log_params({"trigger": reason, "git_commit": git_commit(), "dvc_data_hash": dvc_data_hash(),
                           "data_hash": hashlib.sha1(df.sort_values("clip").to_csv(index=False).encode()).hexdigest()[:12],
                           "n_recordings": len(df), "n_subjects": df.subject.nunique(), "seed": p["train"]["seed"],
                           "n_perm": p["train"]["n_perm"] if n_perm is None else n_perm})
        if not ok:
            decision = {**base, "trained": False, "promote": False, "validated": False,
                        "reasons": [f"insufficient data: {why}"]}
            mlflow.set_tags({"decision": "skipped", "reason": why})
            return decision
        res = evaluate(df, p, n_perm)
        best, thr = res["best"], tier_thresholds(df.v_statures_per_s)
        mlflow.log_metrics({"loso_accuracy": best["accuracy"], "loso_macro_f1": best["f1"],
                            "loso_macro_precision": best["precision"], "loso_macro_recall": best["recall"],
                            "permutation_p": res["perm_p"], "chance_accuracy": res["chance_acc"]})
        mlflow.log_params({"candidate": f"{best['features']}/{best['model']}/aug={best['augment']}"})
        gd = gate_decision(res, champ, p)
        bundle = fit_final(df, best, p)
        meta = {"loso_macro_f1": best["f1"], "loso_accuracy": best["accuracy"], "perm_p": res["perm_p"],
                "validated": gd["validated"], "provisional": gd["bootstrap"] and not gd["validated"],
                "candidate": best, "tier_thresholds_statures_per_s": thr, "n_recordings": len(df),
                "n_subjects": res["n_subjects"], "trained_utc": base["time_utc"], "git_commit": git_commit(),
                "classes": CLASS_NAMES}
        ref = df[["clip", "subject", "label", "v_statures_per_s"] + FEATURES]
        with tempfile.TemporaryDirectory() as td:
            export_champion(bundle, meta, Path(td) / "model", reference=ref)
            (Path(td) / "report.json").write_text(json.dumps(res, indent=2, default=float))
            mlflow.log_artifacts(str(Path(td) / "model"), "model")
            mlflow.log_artifact(str(Path(td) / "report.json"))
        client = client.__class__()
        try:
            client.create_registered_model(MODEL_NAME)
        except Exception:
            pass
        mv = client.create_model_version(MODEL_NAME, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id)
        tags = {"loso_macro_f1": f"{best['f1']:.6f}", "perm_p": f"{res['perm_p']:.4f}",
                "validated": str(gd["validated"]), "promoted": str(gd["promote"])}
        for k, v in tags.items():
            client.set_model_version_tag(MODEL_NAME, mv.version, k, v)
        client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
        if gd["promote"]:
            client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
            meta["version"] = str(mv.version)
            export_champion(bundle, meta, reference=ref)
        mlflow.set_tags({"decision": "promoted" if gd["promote"] else "refused", "model_version": str(mv.version)})
        decision = {**base, "trained": True, "promote": gd["promote"], "validated": gd["validated"],
                    "reasons": gd["reasons"], "model_version": str(mv.version), "candidate": best,
                    "loso_accuracy": best["accuracy"], "loso_macro_f1": best["f1"], "perm_p": res["perm_p"],
                    "chance_acc": res["chance_acc"], "seconds": round(time.time() - t0, 1)}
    return decision
