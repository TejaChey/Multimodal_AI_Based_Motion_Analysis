"""Cross-modal cadence agreement on UTD-MHAD jogging (action 22, 8 subjects x 4 trials).

Replaces the original UTD-MHAD "3-class skill" experiment, whose labels were
tertiles of hip_angle_mean, i.e. one of the input features (circular). Here
UTD-MHAD validates the multimodal measurement layer instead: the stride
frequency from video (MediaPipe knee trajectory) must agree with the stride
frequency from the synchronised thigh-worn inertial sensor (50 Hz).

Output: results/utd_cadence.csv, figures/utd_cadence.png
"""
import re
import warnings

warnings.filterwarnings("ignore")
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mediapipe.python.solutions import pose as mp_pose
from scipy.io import loadmat
from scipy.signal import detrend

from common import FIG, OUT, ROOT

UTD = ROOT / "Phase_1" / "data" / "utd-mhad"
LAG = (0.4, 1.6)  # plausible jogging stride period range (s)


def acf_period(x, fs):
    x = detrend(x)
    x = (x - x.mean()) / (x.std() + 1e-12)
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    a, b = int(LAG[0] * fs), min(int(LAG[1] * fs), len(ac) - 2)
    seg = ac[a:b]
    # first local maximum (avoid picking a multiple of the period)
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    i = peaks[0] if peaks else int(np.argmax(seg))
    return (a + i) / fs


def video_stride(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ys = []
    with mp_pose.Pose(model_complexity=1) as pose:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            r = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            ys.append(r.pose_landmarks.landmark[25].y - r.pose_landmarks.landmark[26].y
                      if r.pose_landmarks else np.nan)
    y = pd.Series(ys).interpolate(limit_direction="both").values
    return 1 / acf_period(y, fps), np.isfinite(np.array(ys, float)).mean()


def imu_stride(path, fs=50.0):
    d = loadmat(path)["d_iner"]
    acc = d[:, :3]
    ax = acc[:, np.argmax(acc.std(0))]
    return 1 / acf_period(ax, fs)


if __name__ == "__main__":
    rows = []
    for v in sorted((UTD / "RGB").glob("a22_*_color.avi")):
        s, t = map(int, re.findall(r"s(\d+)_t(\d+)", v.name)[0])
        fv, det = video_stride(v)
        fi = imu_stride(UTD / "Inertial" / f"a22_s{s}_t{t}_inertial.mat")
        rows.append({"subject": s, "trial": t, "stride_hz_video": fv, "stride_hz_imu": fi, "pose_detect_rate": det})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "utd_cadence.csv", index=False)
    diff = d.stride_hz_video - d.stride_hz_imu
    r = np.corrcoef(d.stride_hz_video, d.stride_hz_imu)[0, 1]
    loa = 1.96 * diff.std()
    print(d.round(3).to_string(index=False))
    print(f"n={len(d)}  MAE={diff.abs().mean():.3f} Hz  bias={diff.mean():+.3f} Hz  "
          f"95% LoA=±{loa:.3f} Hz  r={r:.3f}  within 0.1 Hz: {(diff.abs() < 0.1).mean():.0%}")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].scatter(d.stride_hz_imu, d.stride_hz_video, c=d.subject, cmap="tab10")
    lim = [d[["stride_hz_video", "stride_hz_imu"]].min().min() - .1, d[["stride_hz_video", "stride_hz_imu"]].max().max() + .1]
    ax[0].plot(lim, lim, "k--", lw=.8)
    ax[0].set(xlabel="IMU stride freq (Hz)", ylabel="video stride freq (Hz)", title=f"UTD-MHAD jog (n={len(d)}), r={r:.2f}")
    m = (d.stride_hz_video + d.stride_hz_imu) / 2
    ax[1].scatter(m, diff, c=d.subject, cmap="tab10")
    for yv in (diff.mean(), diff.mean() + loa, diff.mean() - loa):
        ax[1].axhline(yv, ls="--", color="grey", lw=.8)
    ax[1].set(xlabel="mean (Hz)", ylabel="video − IMU (Hz)", title="Bland–Altman")
    fig.tight_layout(); fig.savefig(FIG / "utd_cadence.png", dpi=200)
