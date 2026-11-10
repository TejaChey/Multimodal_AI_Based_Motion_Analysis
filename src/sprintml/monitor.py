"""Drift monitoring: per-feature two-sample KS (Bonferroni family-wise alpha) + PSI.

Alarms are ARMED only when the reference set and monitoring window are large enough for the
test to have useful power (paper power analysis: n_ref >= 100, m >= 30). Below that the monitor
still reports p-values but marks them informational, instead of giving false confidence.
"""
import html
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import art_dir
from .features import FEATURES


def psi(ref, new, bins=10) -> float:
    ref, new = np.asarray(ref, float), np.asarray(new, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    b = np.histogram(new, edges)[0] / len(new) + 1e-4
    return float(np.sum((b - a) * np.log(b / a)))


def drift_report(reference: pd.DataFrame, window: pd.DataFrame, p: dict, features=FEATURES) -> dict:
    m = p["monitor"]
    n_ref, n_win = len(reference), len(window)
    alpha = m["alpha_family"] / len(features)
    rows = []
    for f in features:
        r, w = reference[f].dropna().values, window[f].dropna().values
        if len(r) < 2 or len(w) < 2:
            continue
        ks = ks_2samp(r, w)
        rows.append({"feature": f, "ks_stat": float(ks.statistic), "p_value": float(ks.pvalue),
                     "psi": psi(r, w), "drifted": bool(ks.pvalue < alpha)})
    armed = n_ref >= m["min_reference"] and n_win >= m["min_window"]
    flagged = [r["feature"] for r in rows if r["drifted"]]
    status = ("ALARM" if flagged else "ok") if armed else (
        "informational_alarm_unarmed" if flagged else "insufficient_data")
    return {"time_utc": datetime.now(timezone.utc).isoformat(), "n_reference": n_ref, "n_window": n_win,
            "alpha_per_feature": alpha, "armed": armed,
            "arming_rule": f"n_reference >= {m['min_reference']} and n_window >= {m['min_window']}",
            "status": status, "alarm": bool(armed and flagged), "drifted_features": flagged,
            "max_psi": max((r["psi"] for r in rows), default=0.0),
            "features": sorted(rows, key=lambda r: r["p_value"])}


def write_report(rep: dict, name="drift_report") -> None:
    d = art_dir()
    (d / f"{name}.json").write_text(json.dumps(rep, indent=2))
    hot = ' style="background:#fdd"'
    body = "".join(
        "<tr{}><td>{}</td><td>{:.3f}</td><td>{:.4f}</td><td>{:.3f}</td></tr>".format(
            hot if r["drifted"] else "", html.escape(r["feature"]), r["ks_stat"], r["p_value"], r["psi"])
        for r in rep["features"])
    (d / f"{name}.html").write_text(
        f"<html><body><h2>Drift report: {rep['status']}</h2><p>{rep['time_utc']} | reference n={rep['n_reference']}, "
        f"window n={rep['n_window']} | armed={rep['armed']} ({rep['arming_rule']}) | per-feature alpha="
        f"{rep['alpha_per_feature']:.4f}</p><table border=1 cellpadding=4><tr><th>feature</th><th>KS</th>"
        f"<th>p</th><th>PSI</th></tr>{body}</table></body></html>")


def load_reference() -> pd.DataFrame:
    path = art_dir() / "champion" / "reference_features.csv"
    if not path.exists():
        raise FileNotFoundError("no reference_features.csv: train a model first")
    return pd.read_csv(path)
