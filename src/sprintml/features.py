"""19 sprint biomechanics features (9 video + 10 IMU) and per-recording metadata."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch

from .labels import measure_velocity
from .pose import runner_track

VIDEO = ["knee_angle_mean", "hip_angle_mean", "knee_sym_index", "hip_sym_index", "trunk_lean_deg",
         "vert_osc_stature", "step_freq_video_hz", "gct_ms", "duty_factor"]
IMU = ["peak_accel", "mean_accel", "std_accel", "rms_jerk", "peak_gyro", "mean_gyro", "std_gyro",
       "arm_freq_gyro_hz", "sparc_accel", "sparc_gyro"]
FEATURES = VIDEO + IMU
FS = 100.0


def _angle(a, b, c):
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _P(d, k):
    return d[[f"x{k}", f"y{k}"]].values


def _sym(r, l):
    return abs(r - l) / ((r + l) / 2) * 100


def resample(d: pd.DataFrame, fps: float) -> pd.DataFrame:
    full = pd.DataFrame({"frame": np.arange(d.frame.min(), d.frame.max() + 1)})
    full = full.merge(d, on="frame", how="left").interpolate(limit=6, limit_direction="both")
    full["t"] = full.frame / fps
    return full.dropna(subset=["x23"])


def video_features(pose: pd.DataFrame, p: dict):
    """Returns (features dict, tracked span in seconds, resampled track)."""
    f = p["features"]
    fps = float(pose.fps.iloc[0])
    d = resample(runner_track(pose), fps)
    seg = lambda a, b: np.hypot(d[f"x{a}"] - d[f"x{b}"], d[f"y{a}"] - d[f"y{b}"])
    L = np.median(((seg(11, 23) + seg(23, 25) + seg(25, 27)) + (seg(12, 24) + seg(24, 26) + seg(26, 28))) / 2)
    S = L / p["labels"]["seg_to_stature"]
    hip, sh = (_P(d, 23) + _P(d, 24)) / 2, (_P(d, 11) + _P(d, 12)) / 2
    kneeL, kneeR = _angle(_P(d, 23), _P(d, 25), _P(d, 27)), _angle(_P(d, 24), _P(d, 26), _P(d, 28))
    hipL, hipR = _angle(_P(d, 11), _P(d, 23), _P(d, 25)), _angle(_P(d, 12), _P(d, 24), _P(d, 26))
    lean = np.degrees(np.arctan2(-(sh[:, 0] - hip[:, 0]), hip[:, 1] - sh[:, 1]))
    hy = hip[:, 1] - np.polyval(np.polyfit(hip[:, 0], hip[:, 1], 1), hip[:, 0])
    # a planted foot is stationary in the world frame -> stance when ankle x-speed is small
    w = 7 if len(d) > 9 else 5
    v_run = abs(np.polyfit(d.t, hip[:, 0], 1)[0])
    vxL = np.abs(savgol_filter(d.x27.values, w, 2, deriv=1) * fps)
    vxR = np.abs(savgol_filter(d.x28.values, w, 2, deriv=1) * fps)
    stance = np.minimum(vxL, vxR) < f["stance_speed_fraction"] * v_run
    runs, start = [], None
    for i, s in enumerate(np.append(stance, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append([start, i])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= f["merge_gap_frames"]:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    runs = [r for r in merged if r[1] - r[0] >= f["min_contact_frames"]]
    kept = []
    for r in runs:  # a step cannot start < min_step_interval after the previous one
        if not kept or (r[0] - kept[-1][0]) / fps >= f["min_step_interval_s"]:
            kept.append(r)
    runs = kept
    onsets = np.array([d.t.values[a] for a, _ in runs])
    gct = np.mean([(b - a) / fps for a, b in runs]) * 1000 if runs else np.nan
    step_f = 1 / np.median(np.diff(onsets)) if len(onsets) >= 3 else np.nan
    feats = {
        "knee_angle_mean": float(np.mean((kneeL + kneeR) / 2)),
        "hip_angle_mean": float(np.mean((hipL + hipR) / 2)),
        "knee_sym_index": float(_sym(kneeR.mean(), kneeL.mean())),
        "hip_sym_index": float(_sym(hipR.mean(), hipL.mean())),
        "trunk_lean_deg": float(np.mean(lean)),
        "vert_osc_stature": float(np.std(hy) / S),
        "step_freq_video_hz": float(step_f), "gct_ms": float(gct),
        "duty_factor": float(gct / 1000 * step_f),
    }
    return feats, float(d.t.max() - d.t.min()), d


def _sparc(sig, fs, fc=10.0):
    f, pw = welch(sig - sig.mean(), fs=fs, nperseg=min(256, len(sig)))
    sel = f <= fc
    pw, f = pw[sel] / (pw[sel].max() + 1e-12), f[sel] / fc
    return float(-np.sum(np.sqrt(np.diff(f) ** 2 + np.diff(pw) ** 2)))


def _dom_freq(sig, fs, lo, hi):
    f, pw = welch(sig - sig.mean(), fs=fs, nperseg=min(256, len(sig)), nfft=1024)
    sel = (f >= lo) & (f <= hi)
    return float(f[sel][np.argmax(pw[sel])])


def imu_features(m: pd.DataFrame, win_s: float):
    fs = 1 / np.median(np.diff(m.t))
    acc = np.sqrt(m.x_a ** 2 + m.y_a ** 2 + m.z_a ** 2).values
    gyr = np.sqrt(m.x_g ** 2 + m.y_g ** 2 + m.z_g ** 2).values
    n = int(round(win_s * fs))
    if 10 < n < len(acc):  # highest-intensity window matching the tracked running time
        sm = np.convolve(acc, np.ones(5) / 5, mode="same")
        s = int(np.argmax(np.convolve(sm, np.ones(n), mode="valid")))
        acc, gyr = acc[s:s + n], gyr[s:s + n]
    jerk = np.diff(acc) * fs
    return {"peak_accel": float(acc.max()), "mean_accel": float(acc.mean()), "std_accel": float(acc.std()),
            "rms_jerk": float(np.sqrt(np.mean(jerk ** 2))), "peak_gyro": float(gyr.max()),
            "mean_gyro": float(gyr.mean()), "std_gyro": float(gyr.std()),
            "arm_freq_gyro_hz": _dom_freq(gyr, fs, 1.0, 3.5),
            "sparc_accel": _sparc(acc, fs), "sparc_gyro": _sparc(gyr, fs)}, float(fs)


# ── cross-modal cadence (used by the data-quality gate) ──────────────────────
def _acf_period(x, fs, lo=0.3, hi=0.8):
    x = (x - x.mean()) / (x.std() + 1e-12)
    ac = np.correlate(x, x, "full")[len(x) - 1:] / len(x)
    a, b = int(lo * fs), min(int(hi * fs), len(ac) - 1)
    return np.nan if b <= a else (a + np.argmax(ac[a:b])) / fs


def wrist_cadence_hz(track: pd.DataFrame, fps: float, stature_px: float) -> float:
    """Stride frequency from the wrists' swing relative to the hip (mean of both wrists)."""
    t = track.frame.values / fps
    hipx = (track.x23 + track.x24).values / 2
    tg = np.arange(t[0], t[-1], 1 / FS)
    if len(tg) < 40:
        return float("nan")
    hz = []
    for k in (15, 16):
        rel = np.interp(tg, t, (track[f"x{k}"].values - hipx) / stature_px)
        per = _acf_period(savgol_filter(rel, 15, 3, deriv=1) * FS, FS)
        hz.append(1 / per if per == per else np.nan)
    return float(np.nanmean(hz))


def gyro_cadence_hz(m: pd.DataFrame) -> float:
    tg = np.arange(0, m.t.iloc[-1], 1 / FS)
    gy = {a: np.interp(tg, m.t.values, m[f"{a}_g"].values) for a in "xyz"}
    dom = max("xyz", key=lambda a: gy[a].std())
    per = _acf_period(gy[dom], FS)
    return float(1 / per) if per == per else float("nan")


def extract_features(pose: pd.DataFrame, imu: pd.DataFrame, p: dict) -> dict:
    """Features + label inputs + quality metadata for one recording (pose and IMU already loaded)."""
    vel = measure_velocity(pose, p)
    vf, span, track = video_features(pose, p)
    inf, imu_hz = imu_features(imu, span)
    end = datetime.fromtimestamp(imu.time.iloc[-1] / 1e9, timezone.utc)
    feats = {**vf, **inf}
    meta = {"fps": float(pose.fps.iloc[0]), "imu_hz": imu_hz, "tracked_s": span,
            "wrist_cadence_hz": wrist_cadence_hz(track, float(pose.fps.iloc[0]), vel["stature_px"]),
            "gyro_cadence_hz": gyro_cadence_hz(imu), "imu_end_utc": end.isoformat(),
            **{k: vel[k] for k in ("v_statures_per_s", "v_mps", "fit_r2", "frames_used")}}
    return {"features": feats, "meta": meta}
