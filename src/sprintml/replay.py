"""Cold-start replay: the 9 recordings arrive in real capture order through the real feature store,
retraining trigger and promotion gate (isolated artifacts dir). Demonstrates that the loop works
mechanically; with this few athletes it cannot demonstrate an accuracy improvement."""
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import CLASS_NAMES, art_dir, load_params
from .features import FEATURES
from .labels import assign_tiers, tier_thresholds
from .pipeline import maybe_retrain
from .store import FeatureStore


def run(n_perm=100) -> dict:
    p = load_params()
    p["retrain"]["every_n_new"] = 1
    real = art_dir()
    raw = pd.read_csv(real / "features_raw.csv")
    qual = json.loads((real / "quality_report.json").read_text())
    order = raw.sort_values("video_creation_utc").reset_index(drop=True)
    clean = raw[raw["clip"].map(lambda c: qual[c]["passed"])]
    final_thr = tier_thresholds(clean.v_statures_per_s)
    truth = dict(zip(clean["clip"], assign_tiers(clean.v_statures_per_s, final_thr)))

    steps, prev = [], os.environ.get("SPRINTML_ARTIFACTS")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SPRINTML_ARTIFACTS"] = tmp
        try:
            store = FeatureStore()
            for i, row in order.iterrows():
                clip, q = row["clip"], qual[row["clip"]]
                feats = {k: float(row[k]) for k in FEATURES}
                # prequential: what would the current champion have predicted for this arrival?
                pred = None
                mp = Path(tmp) / "champion" / "model.joblib"
                if mp.exists() and q["passed"]:
                    b = joblib.load(mp)
                    X = b["scaler"].transform(pd.DataFrame([feats])[b["features"]].values)
                    pred = CLASS_NAMES[int(b["clf"].predict(X)[0])]
                store.add(clip, row["subject"], feats, {"passed": q["passed"], "failed": q["failed"], "checks": {}},
                          float(row["v_statures_per_s"]), ts=1.0 + i)
                d = maybe_retrain(p, store, n_perm=n_perm)
                steps.append({"step": i + 1, "arrival": clip, "quality_passed": q["passed"],
                              "quarantined_for": q["failed"], "passed_so_far": store.count(passed_only=True),
                              "champion_prediction_before_training": pred,
                              "true_tier_final_cohort": CLASS_NAMES[int(truth[clip])] if clip in truth else None,
                              "retrain_triggered": d.get("triggered", False),
                              "trained": d.get("trained"), "promoted": d.get("promote"),
                              "validated": d.get("validated"), "reasons": d.get("reasons"),
                              "model_version": d.get("model_version"), "loso_accuracy": d.get("loso_accuracy"),
                              "perm_p": d.get("perm_p")})
                print(f"{i+1}. {clip:18s} quality={'ok ' if q['passed'] else 'BAD'} "
                      f"retrain={d.get('triggered')} promoted={d.get('promote')} {d.get('reasons', '')}", flush=True)
        finally:
            os.environ.pop("SPRINTML_ARTIFACTS", None)
            if prev:
                os.environ["SPRINTML_ARTIFACTS"] = prev
    scored = [s for s in steps if s["champion_prediction_before_training"]]
    out = {"steps": steps, "n_perm": n_perm, "n_prequential_predictions": len(scored),
           "prequential_correct": sum(s["champion_prediction_before_training"] == s["true_tier_final_cohort"]
                                      for s in scored),
           "models_trained": sum(bool(s["trained"]) for s in steps),
           "models_promoted": sum(bool(s["promoted"]) for s in steps)}
    (real / "replay_results.json").write_text(json.dumps(out, indent=2, default=float))
    return out
