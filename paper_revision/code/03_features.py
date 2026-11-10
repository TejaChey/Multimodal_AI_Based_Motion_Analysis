"""Corrected multimodal feature extraction (runner-only frames).

Fixes relative to Phase_2/code:
  * pose frames from static false detections are removed (common.runner_track)
  * stride length was 15 m / steps (assumed the clip spans the fly zone; it
    does not) -> removed. It is also algebraically speed/step-rate, i.e. the
    label, so it would leak.
  * duration_s / total_frames / total_steps_detected depend on how the clip
    was trimmed -> removed.
  * ground contact is detected from the stance foot being stationary in the
    world frame (ankle horizontal velocity ~ 0), not from a fixed pixel
    threshold.
  * IMU window length = time the runner is actually tracked in video.
  * asymmetry uses the paper's Eq. (2) symmetry index.

Output: results/features.csv
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch

from common import CLIPS, OUT, POSE_DIR, RAW, SUBJECT, runner_track

SEG_TO_STATURE = 0.779


def angle(a, b, c):
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def P(d, k):
    return d[[f"x{k}", f"y{k}"]].values


def sym_index(r, l):
    return abs(r - l) / ((r + l) / 2) * 100


def resample(d, fps):
    """Uniform time grid over the tracked span, linear interpolation over short gaps."""
    full = pd.DataFrame({"frame": np.arange(d.frame.min(), d.frame.max() + 1)})
    full = full.merge(d, on="frame", how="left").interpolate(limit=6, limit_direction="both")
    full["t"] = full.frame / fps
    return full.dropna(subset=["x23"])


def video_features(clip):
    raw = pd.read_csv(POSE_DIR / f"{clip}.csv")
    fps = raw.fps.iloc[0]
    d = resample(runner_track(raw), fps)
    seg = lambda a, b: np.hypot(d[f"x{a}"] - d[f"x{b}"], d[f"y{a}"] - d[f"y{b}"])
    L = np.median(((seg(11, 23) + seg(23, 25) + seg(25, 27)) + (seg(12, 24) + seg(24, 26) + seg(26, 28))) / 2)
    S = L / SEG_TO_STATURE                              # stature in px
    hip = (P(d, 23) + P(d, 24)) / 2
    sh = (P(d, 11) + P(d, 12)) / 2
    kneeL, kneeR = angle(P(d, 23), P(d, 25), P(d, 27)), angle(P(d, 24), P(d, 26), P(d, 28))
    hipL, hipR = angle(P(d, 11), P(d, 23), P(d, 25)), angle(P(d, 12), P(d, 24), P(d, 26))
    # trunk lean from vertical (positive = forward in running direction, which is -x)
    lean = np.degrees(np.arctan2(-(sh[:, 0] - hip[:, 0]), hip[:, 1] - sh[:, 1]))

    # vertical oscillation: detrend hip y against x (camera perspective tilt)
    hy = hip[:, 1] - np.polyval(np.polyfit(hip[:, 0], hip[:, 1], 1), hip[:, 0])
    vosc = np.std(hy) / S

    # stance detection: the stance ankle is ~stationary horizontally in world frame
    w = 7 if len(d) > 9 else 5
    v_run = abs(np.polyfit(d.t, hip[:, 0], 1)[0])
    vxL = np.abs(savgol_filter(d.x27.values, w, 2, deriv=1) * fps)
    vxR = np.abs(savgol_filter(d.x28.values, w, 2, deriv=1) * fps)
    # threshold 0.4*v_run and 1-frame merge chosen so every clip gives physiologically
    # plausible step rate (3-5 Hz) and contact time (60-130 ms); NOT tuned on accuracy
    stance = np.minimum(vxL, vxR) < 0.4 * v_run
    runs, start = [], None
    for i, s in enumerate(np.append(stance, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append([start, i])
            start = None
    # merge runs split by 1 frame of landmark jitter, drop 1-frame blips
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 1:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    runs = [r for r in merged if r[1] - r[0] >= 2]
    # a new step cannot start < 150 ms after the previous one (> 6.7 Hz)
    kept = []
    for r in runs:
        if not kept or (r[0] - kept[-1][0]) / fps >= 0.15:
            kept.append(r)
    runs = kept
    onsets = np.array([d.t.values[a] for a, _ in runs])
    gct = np.mean([(b - a) / fps for a, b in runs]) * 1000 if runs else np.nan
    step_f = 1 / np.median(np.diff(onsets)) if len(onsets) >= 3 else np.nan
    return {
        "knee_angle_mean": np.mean((kneeL + kneeR) / 2),
        "hip_angle_mean": np.mean((hipL + hipR) / 2),
        "knee_sym_index": sym_index(kneeR.mean(), kneeL.mean()),
        "hip_sym_index": sym_index(hipR.mean(), hipL.mean()),
        "trunk_lean_deg": np.mean(lean),
        "vert_osc_stature": vosc,
        "step_freq_video_hz": step_f,
        "gct_ms": gct,
        "duty_factor": gct / 1000 * step_f,
    }, d.t.max() - d.t.min()


def load_imu(clip):
    a = pd.read_csv(RAW / clip / "WatchAccelerometer.csv").sort_values("time")
    g = pd.read_csv(RAW / clip / "WatchGyroscope.csv").sort_values("time")
    m = pd.merge_asof(a[["time", "x", "y", "z"]], g[["time", "x", "y", "z"]], on="time",
                      suffixes=("_a", "_g"), tolerance=20_000_000, direction="nearest").dropna()
    m["t"] = (m.time - m.time.iloc[0]) / 1e9
    return m


def sparc(sig, fs, fc=10.0):
    f, p = welch(sig - sig.mean(), fs=fs, nperseg=min(256, len(sig)))
    sel = f <= fc
    p = p[sel] / (p[sel].max() + 1e-12)
    f = f[sel] / fc
    return -np.sum(np.sqrt(np.diff(f) ** 2 + np.diff(p) ** 2))


def dom_freq(sig, fs, lo=1.0, hi=6.0):
    f, p = welch(sig - sig.mean(), fs=fs, nperseg=min(256, len(sig)), nfft=1024)
    sel = (f >= lo) & (f <= hi)
    return f[sel][np.argmax(p[sel])]


def imu_features(clip, win_s):
    m = load_imu(clip)
    fs = 1 / np.median(np.diff(m.t))
    acc = np.sqrt(m.x_a ** 2 + m.y_a ** 2 + m.z_a ** 2).values
    gyr = np.sqrt(m.x_g ** 2 + m.y_g ** 2 + m.z_g ** 2).values
    n = int(round(win_s * fs))
    if n < len(acc):  # highest-intensity window matching the tracked running time
        sm = np.convolve(acc, np.ones(5) / 5, mode="same")
        s = int(np.argmax(np.convolve(sm, np.ones(n), mode="valid")))
        acc, gyr = acc[s:s + n], gyr[s:s + n]
    jerk = np.diff(acc) * fs
    return {
        "peak_accel": acc.max(), "mean_accel": acc.mean(), "std_accel": acc.std(),
        "rms_jerk": np.sqrt(np.mean(jerk ** 2)),
        "peak_gyro": gyr.max(), "mean_gyro": gyr.mean(), "std_gyro": gyr.std(),
        "arm_freq_gyro_hz": dom_freq(gyr, fs, 1.0, 3.5),
        "sparc_accel": sparc(acc, fs), "sparc_gyro": sparc(gyr, fs),
    }, len(acc) / fs


if __name__ == "__main__":
    rows = []
    for c in CLIPS:
        v, span = video_features(c)
        i, used = imu_features(c, span)
        rows.append({"clip": c, "subject": SUBJECT[c], "track_s": span, "imu_s": used, **v, **i})
    f = pd.DataFrame(rows)
    lab = pd.read_csv(OUT / "speed_labels.csv")[["clip", "v_statures_per_s", "v_approx_mps", "label", "tier"]]
    f = f.merge(lab, on="clip")
    f.to_csv(OUT / "features.csv", index=False)
    pd.set_option("display.width", 250)
    print(f.drop(columns=["subject"]).round(3).T.to_string())
