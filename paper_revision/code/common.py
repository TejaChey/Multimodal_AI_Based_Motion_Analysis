"""Shared paths and metadata for the camera-ready revision experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Phase_2" / "data" / "sprint_raw"
REV = ROOT / "paper_revision"
OUT = REV / "results"
POSE_DIR = OUT / "pose"
FIG = REV / "figures"
for d in (OUT, POSE_DIR, FIG):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Recording folder -> athlete identity. Yeshwanth ran two trials, so there are
# 9 recordings from 8 athletes; every subject-independent split groups by this.
SUBJECT = {
    "Amith": "S1", "Chandra1": "S2", "Hanshi2": "S3", "Jeeva": "S4",
    "Mohammed_araar": "S5", "Sachin": "S6", "Tejas": "S7",
    "Yeshwanth": "S8", "Yeshwanth_trail_2": "S8",
}
CLIPS = sorted(SUBJECT)

CLASS_NAMES = ["Beginner", "Intermediate", "Elite"]


def runner_track(df, max_gap=6, max_jump_px=60, size_ratio=1.5):
    """Keep only the frames belonging to the sprinting athlete.

    MediaPipe Pose sometimes locks onto a static false detection (stands,
    bystander) before the runner enters the frame. Split detected frames into
    continuous tracks (no long gaps, no position jumps, stable body size) and
    keep the track with the largest horizontal travel.
    """
    import numpy as np
    d = df[df.detected].copy()
    hx = ((d.x23 + d.x24) / 2).values
    size = np.hypot(d.x11 - d.x23, d.y11 - d.y23).values
    fr = d.frame.values
    seg_id, cur = [0], 0
    for i in range(1, len(d)):
        gap = fr[i] - fr[i - 1]
        jump = abs(hx[i] - hx[i - 1]) / gap
        ratio = max(size[i], size[i - 1]) / max(min(size[i], size[i - 1]), 1e-6)
        if gap > max_gap or jump > max_jump_px or ratio > size_ratio:
            cur += 1
        seg_id.append(cur)
    d["seg"] = seg_id
    travel = d.groupby("seg").apply(lambda g: ((g.x23 + g.x24) / 2).agg(np.ptp))
    return d[d.seg == travel.idxmax()].drop(columns="seg")


def video_path(clip: str) -> Path:
    return next(p for p in (RAW / clip).iterdir() if p.suffix.lower() == ".mp4")
