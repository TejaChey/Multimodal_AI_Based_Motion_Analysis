"""Batch pipeline stages (DVC): pose -> features -> validate -> train. Plus retraining and the monitor loop."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import labels, monitor, quality
from .config import art_dir, load_params, raw_dir
from .imu import load_imu
from .pose import extract_pose, video_meta
from .store import FeatureStore
from .train import current_champion, train_and_register


def recordings():
    out = {}
    for d in sorted(p for p in raw_dir().iterdir() if p.is_dir()):
        vids = [f for f in d.iterdir() if f.suffix.lower() == ".mp4"]
        if vids and (d / "WatchAccelerometer.csv").exists() and (d / "WatchGyroscope.csv").exists():
            out[d.name] = {"video": vids[0], "accel": d / "WatchAccelerometer.csv", "gyro": d / "WatchGyroscope.csv"}
    return out


def stage_pose(p):
    out = art_dir() / "pose"
    out.mkdir(exist_ok=True, parents=True)
    for clip, r in recordings().items():
        df = extract_pose(r["video"], p)
        df.to_csv(out / f"{clip}.csv", index=False)
        (out / f"{clip}.meta.json").write_text(json.dumps(video_meta(r["video"])))
        print(f"pose {clip}: {len(df)} frames, detected {df.detected.mean():.0%}", flush=True)


def stage_features(p):
    rows, rec = [], recordings()
    for clip, r in rec.items():
        pose = pd.read_csv(art_dir() / "pose" / f"{clip}.csv")
        vm = json.loads((art_dir() / "pose" / f"{clip}.meta.json").read_text())
        x = F.extract_features(pose, load_imu(r["accel"], r["gyro"]), p)
        x["meta"]["video_creation_utc"] = vm.get("creation_utc")
        rows.append({"clip": clip, "subject": p["subjects"].get(clip, clip), **x["features"], **x["meta"]})
    pd.DataFrame(rows).to_csv(art_dir() / "features_raw.csv", index=False)
    print(f"features: {len(rows)} recordings")


def stage_validate(p):
    raw = pd.read_csv(art_dir() / "features_raw.csv")
    recs = []
    for _, row in raw.iterrows():
        feats, meta = {k: row[k] for k in F.FEATURES}, row.to_dict()
        recs.append({"clip": row["clip"], "meta": meta, "quality": quality.check_recording(feats, meta, p)})
    quality.apply_clock_check(recs, p)
    ok = [r["clip"] for r in recs if r["quality"]["passed"]]
    report = {r["clip"]: {"passed": r["quality"]["passed"], "failed": r["quality"]["failed"],
                          "checks": r["quality"]["checks"]} for r in recs}
    (art_dir() / "quality_report.json").write_text(json.dumps(report, indent=2))
    raw[raw["clip"].isin(ok)].to_csv(art_dir() / "features_clean.csv", index=False)
    for r in recs:
        print(f"quality {r['clip']:20s} {'PASS' if r['quality']['passed'] else 'QUARANTINED ' + str(r['quality']['failed'])}")
    print(f"validate: {len(ok)}/{len(recs)} recordings pass")


def label_frame(df):
    df = df.copy()
    df["label"] = labels.assign_tiers(df.v_statures_per_s, labels.tier_thresholds(df.v_statures_per_s))
    return df


def stage_train(p, reason="pipeline", n_perm=None):
    d = train_and_register(label_frame(pd.read_csv(art_dir() / "features_clean.csv")), p, reason, n_perm)
    (art_dir() / "train_report.json").write_text(json.dumps(d, indent=2, default=float))
    log_event(d)
    print(json.dumps({k: d.get(k) for k in ("promote", "validated", "reasons", "loso_accuracy", "perm_p")}, default=float))
    return d


# ── retraining + monitoring loop ─────────────────────────────────────────────
def log_event(d):
    with open(art_dir() / "retrain_log.jsonl", "a") as f:
        f.write(json.dumps(d, default=float) + "\n")


def _state():
    s = art_dir() / "state.json"
    return json.loads(s.read_text()) if s.exists() else {"trained_on_new": 0}


def training_frame(store: FeatureStore) -> pd.DataFrame:
    """Base dataset + quality-passed recordings collected by the service, labelled automatically."""
    base = pd.read_csv(art_dir() / "features_clean.csv") if (art_dir() / "features_clean.csv").exists() else pd.DataFrame()
    new = store.frame(passed_only=True)
    if len(new):
        new = new.rename(columns={"recording_id": "clip", "athlete_id": "subject"})
        new = new[["clip", "subject", "v_statures_per_s"] + F.FEATURES]
    df = pd.concat([base[["clip", "subject", "v_statures_per_s"] + F.FEATURES] if len(base) else None, new if len(new) else None])
    return label_frame(df.drop_duplicates("clip", keep="last").reset_index(drop=True))


def maybe_retrain(p, store=None, force=False, n_perm=None) -> dict:
    """Triggers: manual (force), armed drift alarm, or every N new quality-passed recordings."""
    store, st = store or FeatureStore(), _state()
    n_new = store.count(passed_only=True)
    pending = n_new - st["trained_on_new"]
    reason = "manual" if force else None
    if not reason:
        try:
            win = store.frame(last=p["monitor"]["window"])
            if len(win):
                rep = monitor.drift_report(monitor.load_reference(), win, p)
                monitor.write_report(rep)
                if rep["alarm"]:
                    reason = "drift_alarm"
        except FileNotFoundError:
            pass
    if not reason and pending >= p["retrain"]["every_n_new"]:
        reason = f"new_data_{pending}"
    if not reason:
        return {"triggered": False, "pending_new_recordings": pending}
    d = train_and_register(training_frame(store), p, reason, n_perm)
    d["triggered"] = True
    (art_dir() / "state.json").write_text(json.dumps({"trained_on_new": n_new}))
    log_event(d)
    return d


def run_monitor(p) -> dict:
    store = FeatureStore()
    win = store.frame(last=p["monitor"]["window"])
    if win.empty:
        return {"status": "no_requests_yet"}
    rep = monitor.drift_report(monitor.load_reference(), win, p)
    monitor.write_report(rep)
    return rep


def loop(p, interval_s=300, metrics_port=9101):
    from prometheus_client import Counter, Gauge, start_http_server
    from .alerts import notify
    g_alarm = Gauge("sprintml_drift_alarm", "1 if an ARMED drift alarm is active")
    g_armed = Gauge("sprintml_drift_armed", "1 if the drift monitor is armed (reference/window large enough)")
    g_p = Gauge("sprintml_drift_min_p_value", "Smallest per-feature KS p-value in the current window")
    g_win = Gauge("sprintml_drift_window_size", "Requests in the monitoring window")
    g_ref = Gauge("sprintml_drift_reference_size", "Reference recordings")
    g_new = Gauge("sprintml_new_recordings_pending", "Quality-passed recordings since the last retrain")
    c_rt = Counter("sprintml_retrain_total", "Retraining runs", ["outcome"])
    start_http_server(metrics_port)
    print(f"monitor/retrain loop every {interval_s}s; metrics on :{metrics_port}", flush=True)
    while True:
        try:
            rep = run_monitor(p)
            if "features" in rep and rep["features"]:
                g_alarm.set(int(rep["alarm"])); g_armed.set(int(rep["armed"]))
                g_p.set(min(f["p_value"] for f in rep["features"]))
                g_win.set(rep["n_window"]); g_ref.set(rep["n_reference"])
                if rep["alarm"]:
                    notify(f"Drift ALARM on {', '.join(rep['drifted_features'])} (window n={rep['n_window']})")
            d = maybe_retrain(p)
            g_new.set(d.get("pending_new_recordings", 0))
            if d.get("triggered"):
                outcome = "promoted" if d.get("promote") else ("refused" if d.get("trained") else "skipped")
                c_rt.labels(outcome).inc()
                notify(f"Retraining ({d.get('reason')}): {outcome}. {'; '.join(d.get('reasons') or [])}")
            print(time.strftime("%H:%M:%S"), "drift:", rep.get("status"), "| retrain:", d.get("triggered"),
                  d.get("reasons", ""), flush=True)
        except Exception as e:  # keep the worker alive
            print("loop error:", repr(e), flush=True)
        time.sleep(interval_s)
