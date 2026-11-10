"""Validate video <-> smartwatch integration (reviewer comment 5).

1. Clock check: watch sample timestamps (UTC, ns) vs. the video container
   creation_time. A consistent offset across recordings indicates the two
   files belong to the same sprint; outliers flag mismatched files.
2. Cadence agreement: stride frequency from video (ankle contacts / 2 and
   wrist swing) vs. from the watch gyroscope (dominant arm-swing period).
3. Waveform agreement: normalised cross-correlation between the video wrist
   velocity (relative to hip) and the watch's dominant gyroscope axis.

Output: results/sync_validation.csv, figures/sync_validation.png
"""
import json
import subprocess
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from common import CLIPS, FIG, OUT, POSE_DIR, RAW, runner_track, video_path

FS = 100.0


def creation_time(clip):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json",
                          str(video_path(clip))], capture_output=True, text=True).stdout
    tags = json.loads(out)["format"]["tags"]
    return datetime.fromisoformat(tags["creation_time"].replace("Z", "+00:00")), tags


def acf_period(x, fs, lo=0.3, hi=0.8):
    """Stride period from the first autocorrelation peak in [lo, hi] s."""
    x = (x - x.mean()) / (x.std() + 1e-12)
    ac = np.correlate(x, x, "full")[len(x) - 1:] / len(x)
    a, b = int(lo * fs), min(int(hi * fs), len(ac) - 1)
    if b <= a:
        return np.nan
    return (a + np.argmax(ac[a:b])) / fs


def wrist_signal(clip):
    raw = pd.read_csv(POSE_DIR / f"{clip}.csv")
    fps = raw.fps.iloc[0]
    d = runner_track(raw)
    full = pd.DataFrame({"frame": np.arange(d.frame.min(), d.frame.max() + 1)}).merge(d, on="frame", how="left")
    full = full.interpolate(limit=6, limit_direction="both").dropna(subset=["x23"])
    t = full.frame.values / fps
    hipx = (full.x23 + full.x24).values / 2
    S = np.median(np.hypot(full.x11 - full.x23, full.y11 - full.y23)) / 0.288  # trunk ~0.288 stature
    tg = np.arange(t[0], t[-1], 1 / FS)
    sigs = {}
    for k in (15, 16):
        rel = np.interp(tg, t, (full[f"x{k}"].values - hipx) / S)
        sigs[k] = savgol_filter(rel, 15, 3, deriv=1) * FS  # statures/s
    return tg - tg[0], sigs


def gyro(clip):
    g = pd.read_csv(RAW / clip / "WatchGyroscope.csv").sort_values("time")
    t = (g.time.values - g.time.values[0]) / 1e9
    tg = np.arange(0, t[-1], 1 / FS)
    return tg, {ax: np.interp(tg, t, g[ax].values) for ax in "xyz"}, g.time.values


def xcorr_best(v, imu):
    """Slide the (shorter) video signal across the IMU signal; max |Pearson r|."""
    n = len(v)
    if len(imu) < n:  # IMU shorter than video track: slide IMU across video instead
        v, imu, n = imu, v, len(imu)
    vz = (v - v.mean()) / v.std()
    best = (0, 0)
    for s in range(len(imu) - n + 1):
        w = imu[s:s + n]
        r = float(np.mean(vz * (w - w.mean()) / (w.std() + 1e-12)))
        if abs(r) > abs(best[0]):
            best = (r, s / FS)
    return best


if __name__ == "__main__":
    feats = pd.read_csv(OUT / "features.csv").set_index("clip")
    rows = []
    for c in CLIPS:
        ct, tags = creation_time(c)
        tv, wr = wrist_signal(c)
        tgy, gy, tns = gyro(c)
        dom = max("xyz", key=lambda a: gy[a].std())
        imu_end = datetime.fromtimestamp(tns[-1] / 1e9, timezone.utc)
        best = max(((k, *xcorr_best(wr[k], gy[dom])) for k in wr), key=lambda z: abs(z[1]))
        rows.append({
            "clip": c, "device": tags.get("com.android.version") and "Android (Samsung)" or "unknown",
            "capture_fps": tags.get("com.android.capture.fps"),
            "video_creation_utc": ct.isoformat(), "imu_end_utc": imu_end.isoformat(),
            "clock_offset_s": (ct - imu_end).total_seconds(),
            "stride_hz_video_contacts": feats.loc[c, "step_freq_video_hz"] / 2,
            "stride_hz_video_wrist": 1 / acf_period(wr[best[0]], FS),
            "stride_hz_watch_gyro": 1 / acf_period(gy[dom], FS),
            "wrist_landmark": {15: "left", 16: "right"}[best[0]],
            "xcorr_r": best[1], "xcorr_lag_s": best[2],
        })
    r = pd.DataFrame(rows)
    med = r.clock_offset_s.median()
    r["clock_consistent"] = (r.clock_offset_s - med).abs() < 5
    r.to_csv(OUT / "sync_validation.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.drop(columns=["video_creation_utc", "imu_end_utc"]).round(3).to_string(index=False))
    ok = r[r.clock_consistent]
    for col in ("stride_hz_video_contacts", "stride_hz_video_wrist"):
        diff = ok[col] - ok.stride_hz_watch_gyro
        print(f"{col} vs watch (clock-consistent n={len(ok)}): MAE={diff.abs().mean():.3f} Hz, "
              f"bias={diff.mean():+.3f} Hz, r={np.corrcoef(ok[col], ok.stride_hz_watch_gyro)[0,1]:.2f}")
    print(f"median clock offset (video creation - watch end) = {med:.2f} s; "
          f"inconsistent: {list(r.loc[~r.clock_consistent, 'clip'])}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    col = np.where(r.clock_consistent, "#1b9e77", "#d95f02")
    ax[0].scatter(r.stride_hz_watch_gyro, r.stride_hz_video_contacts, c=col, label="video: foot contacts")
    ax[0].scatter(r.stride_hz_watch_gyro, r.stride_hz_video_wrist, c=col, marker="x", label="video: wrist swing")
    lim = [1.4, 2.8]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set(xlim=lim, ylim=lim, xlabel="watch gyroscope stride freq (Hz)",
              ylabel="video stride freq (Hz)", title="Cadence agreement")
    ax[0].legend(fontsize=7)
    ax[1].bar(r["clip"], r.clock_offset_s, color=col)
    ax[1].axhline(med, ls="--", color="grey")
    ax[1].set(ylabel="video creation − watch last sample (s)", title="Clock consistency")
    ax[1].tick_params(axis="x", rotation=60, labelsize=7)
    ax[1].set_yscale("symlog")
    fig.tight_layout()
    fig.savefig(FIG / "sync_validation.png", dpi=200)
