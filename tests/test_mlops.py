"""Drift monitor, training protocol (no leakage), promotion gate, registry, feature store."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_table
from sprintml import monitor, train
from sprintml.features import FEATURES
from sprintml.labels import assign_tiers, tier_thresholds
from sprintml.store import FeatureStore


def frame(n, rng, shift=0.0, feat=0):
    X = rng.normal(size=(n, len(FEATURES)))
    X[:, feat] += shift
    return pd.DataFrame(X, columns=FEATURES)


# ── drift monitor ────────────────────────────────────────────────────────────
def test_monitor_false_alarm_rate_is_controlled(params):
    rng, alarms, reps = np.random.default_rng(1), 0, 300
    ref = frame(100, rng)
    for _ in range(reps):
        alarms += monitor.drift_report(ref, frame(30, rng), params)["alarm"]
    assert alarms / reps <= 0.10          # nominal family-wise 5%


def test_monitor_detects_one_sd_shift_when_armed(params):
    rng, hits, reps = np.random.default_rng(2), 0, 100
    ref = frame(100, rng)
    for _ in range(reps):
        hits += monitor.drift_report(ref, frame(30, rng, shift=1.0), params)["alarm"]
    assert hits / reps >= 0.65            # paper power analysis: ~0.84


def test_monitor_is_disarmed_for_small_reference(params):
    rng = np.random.default_rng(3)
    rep = monitor.drift_report(frame(9, rng), frame(30, rng, shift=3.0), params)
    assert not rep["armed"] and not rep["alarm"]
    assert rep["status"] == "informational_alarm_unarmed" and rep["drifted_features"]


def test_psi_flags_distribution_shift():
    rng = np.random.default_rng(4)
    a = rng.normal(size=500)
    assert monitor.psi(a, rng.normal(size=500)) < 0.1
    assert monitor.psi(a, rng.normal(2, 1, 500)) > 0.25


def test_drift_report_written(params, tmp_path, monkeypatch):
    monkeypatch.setenv("SPRINTML_ARTIFACTS", str(tmp_path))
    rng = np.random.default_rng(5)
    monitor.write_report(monitor.drift_report(frame(120, rng), frame(40, rng), params))
    assert json.loads((tmp_path / "drift_report.json").read_text())["armed"]
    assert "Drift report" in (tmp_path / "drift_report.html").read_text()


# ── training protocol ────────────────────────────────────────────────────────
def labelled(df):
    df = df.copy()
    df["label"] = assign_tiers(df.v_statures_per_s, tier_thresholds(df.v_statures_per_s))
    return df


def test_loso_has_no_label_leakage(params):
    """A held-out athlete's prediction must not depend on that athlete's own label."""
    df = labelled(synthetic_table(3, seed=7, separable=False))
    cand = {"features": "fused", "model": "logreg", "augment": True}
    X, y, s = df[FEATURES].values, df.label.values, df.subject.values
    base = train.loso_predict(X, y, s, cand, params, seed=11)
    target = s[0]
    y2 = y.copy()
    y2[s == target] = (y2[s == target] + 1) % 3
    # only the held-out athlete's label changes -> its fold's training set is identical
    changed = train.loso_predict(X, y, s, cand, params, seed=11, y_fit=y2)
    assert base[s == target].tolist() == changed[s == target].tolist()


def test_augmentation_uses_training_fold_only(params):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 4))
    y = np.array([0, 0, 1, 1, 2, 2])
    Xa, ya = train.augment(X, y, np.random.default_rng(1), n=50, noise=0.05)
    assert len(Xa) == 150 and np.bincount(ya).tolist() == [50, 50, 50]
    # synthetic points stay inside the convex hull (+ small jitter) of their own class
    for c in range(3):
        cls, orig = Xa[ya == c], X[y == c]
        assert (cls.min(0) >= orig.min(0) - 0.5).all() and (cls.max(0) <= orig.max(0) + 0.5).all()


def test_evaluate_finds_signal_and_rejects_noise(params):
    sig = train.evaluate(labelled(synthetic_table(3, seed=1, separable=True)), params, n_perm=40, n_jobs=1)
    assert sig["best"]["accuracy"] >= 0.85 and sig["perm_p"] < 0.05
    noise = train.evaluate(labelled(synthetic_table(3, seed=1, separable=False)), params, n_perm=40, n_jobs=1)
    assert noise["perm_p"] > 0.05           # no signal -> no significance (selection-corrected)


def test_trainable_checks(params):
    df = labelled(synthetic_table(3, seed=2))
    assert train.check_trainable(df, params)[0]
    assert not train.check_trainable(df[df.label != 0], params)[0]            # a tier is missing
    assert not train.check_trainable(df.groupby("label").head(1), params)[0]  # < 2 per tier


# ── promotion gate (pure logic) ──────────────────────────────────────────────
def cand(f1, p):
    return {"best": {"f1": f1}, "perm_p": p}


def test_gate_bootstraps_provisional_champion(params):
    d = train.gate_decision(cand(0.3, 0.6), None, params)
    assert d["promote"] and d["bootstrap"] and not d["validated"]


def test_gate_refuses_non_significant_challenger(params):
    d = train.gate_decision(cand(0.9, 0.20), {"loso_macro_f1": "0.3"}, params)
    assert not d["promote"] and "above chance" in d["reasons"][0]


def test_gate_refuses_regression(params):
    d = train.gate_decision(cand(0.4, 0.01), {"loso_macro_f1": "0.7"}, params)
    assert not d["promote"] and any("macro-F1" in r for r in d["reasons"])


def test_gate_promotes_better_validated_challenger(params):
    d = train.gate_decision(cand(0.8, 0.01), {"loso_macro_f1": "0.7"}, params)
    assert d["promote"] and d["validated"] and not d["reasons"]


# ── registry round trip (MLflow + export + restore) ──────────────────────────
@pytest.fixture
def art(tmp_path, monkeypatch):
    monkeypatch.setenv("SPRINTML_ARTIFACTS", str(tmp_path))
    return tmp_path


def test_registry_promotion_refusal_and_restore(params, art):
    good = labelled(synthetic_table(3, seed=1, separable=True))
    d1 = train.train_and_register(good, params, "test-bootstrap", n_perm=40)
    assert d1["promote"] and d1["validated"] and d1["model_version"] == "1"
    meta = json.loads((art / "champion" / "meta.json").read_text())
    assert meta["version"] == "1" and (art / "champion" / "reference_features.csv").exists()

    noise = labelled(synthetic_table(3, seed=1, separable=False))
    d2 = train.train_and_register(noise, params, "test-noise", n_perm=40)
    assert d2["trained"] and not d2["promote"]                       # gate refuses
    assert json.loads((art / "champion" / "meta.json").read_text())["version"] == "1"
    assert train.current_champion()["version"] == "1"

    shutil.rmtree(art / "champion")                                  # e.g. cleared by `dvc repro`
    assert train.restore_champion()
    restored = json.loads((art / "champion" / "meta.json").read_text())
    assert restored["version"] == "1" and (art / "champion" / "model.joblib").exists()


def test_training_skipped_when_data_insufficient(params, art):
    tiny = labelled(synthetic_table(1, seed=1))
    d = train.train_and_register(tiny, params, "test-tiny", n_perm=10)
    assert not d["trained"] and not d["promote"] and "insufficient data" in d["reasons"][0]
    assert not (art / "champion").exists()


# ── feature store ────────────────────────────────────────────────────────────
def test_feature_store_round_trip(art):
    s = FeatureStore()
    q = {"passed": True, "failed": [], "checks": {}}
    for i in range(5):
        s.add(f"r{i}", f"S{i}", {f: float(i) for f in FEATURES}, dict(q, passed=i != 2), 4.0 + i / 10)
    assert s.count() == 5 and s.count(passed_only=True) == 4
    df = s.frame()
    assert len(df) == 4 and "r2" not in df.recording_id.tolist() and df[FEATURES[0]].tolist() == [0, 1, 3, 4]
    assert len(s.frame(last=2)) == 2
