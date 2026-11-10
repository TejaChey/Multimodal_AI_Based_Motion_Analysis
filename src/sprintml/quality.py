"""Automatic data-quality gate. A recording that fails is quarantined, never silently trained on."""
from datetime import datetime

import numpy as np

from .features import FEATURES


def check_recording(features: dict, meta: dict, p: dict) -> dict:
    q, checks = p["quality"], {}

    def add(name, ok, value, limit):
        checks[name] = {"ok": bool(ok), "value": None if value is None else round(float(value), 4), "limit": limit}

    add("video_fps", meta["fps"] >= q["min_fps"], meta["fps"], f">= {q['min_fps']}")
    lo, hi = q["imu_hz_range"]
    add("imu_rate_hz", lo <= meta["imu_hz"] <= hi, meta["imu_hz"], f"{lo}-{hi}")
    add("runner_track_r2", meta["fit_r2"] >= q["min_track_r2"], meta["fit_r2"], f">= {q['min_track_r2']}")
    add("tracked_frames", meta["frames_used"] >= q["min_tracked_frames"], meta["frames_used"],
        f">= {q['min_tracked_frames']}")
    add("features_finite", all(np.isfinite(features.get(k, np.nan)) for k in FEATURES),
        sum(not np.isfinite(features.get(k, np.nan)) for k in FEATURES), "0 missing")
    w, g = meta.get("wrist_cadence_hz"), meta.get("gyro_cadence_hz")
    diff = abs(w - g) if w is not None and g is not None and np.isfinite(w) and np.isfinite(g) else np.nan
    add("video_imu_cadence_agreement_hz", np.isfinite(diff) and diff <= q["max_cadence_diff_hz"],
        diff if np.isfinite(diff) else None, f"<= {q['max_cadence_diff_hz']}")
    return _summary(checks)


def _summary(checks: dict) -> dict:
    failed = [k for k, v in checks.items() if not v["ok"]]
    return {"passed": not failed, "failed": failed, "checks": checks}


def clock_offset_s(meta: dict):
    """video container creation time minus watch last sample (seconds), if both are known."""
    if not meta.get("video_creation_utc") or not meta.get("imu_end_utc"):
        return None
    v = datetime.fromisoformat(meta["video_creation_utc"].replace("Z", "+00:00"))
    i = datetime.fromisoformat(meta["imu_end_utc"])
    return (v - i).total_seconds()


def apply_clock_check(records: list, p: dict) -> None:
    """Cohort check: all recordings of one session share a consistent video-watch clock offset.
    A large deviation means the watch file was not recorded during the filmed sprint."""
    offs = [(r, clock_offset_s(r["meta"])) for r in records]
    known = [o for _, o in offs if o is not None]
    if len(known) < 3:
        return
    med, lim = float(np.median(known)), p["quality"]["max_clock_dev_s"]
    for r, o in offs:
        if o is None:
            continue
        r["quality"]["checks"]["clock_offset_consistency_s"] = {
            "ok": abs(o - med) <= lim, "value": round(o - med, 2), "limit": f"|dev from median| <= {lim}"}
        r["quality"] = _summary(r["quality"]["checks"])
