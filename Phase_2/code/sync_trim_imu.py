"""
sync_trim_imu.py
----------------
Trims IMU data (4s+) to match video duration (1-2s) by detecting the
highest-intensity sprint window using a sliding window on accel magnitude.

Usage:
    # Single athlete folder:
    python Phase_2/code/sync_trim_imu.py --folder Phase_2/data/sprint_raw/Amith

    # Batch process ALL athletes:
    python Phase_2/code/sync_trim_imu.py --batch Phase_2/data/sprint_raw
"""

import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("Phase_2/results/synced_imu")
ACCEL_FILE   = "WatchAccelerometer.csv"
GYRO_FILE    = "WatchGyroscope.csv"
VIDEO_EXTS   = {".mp4", ".mov", ".avi", ".mkv"}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_video_duration(folder: Path) -> float:
    """Return video duration in seconds. Raises if no video found."""
    for f in folder.iterdir():
        if f.suffix.lower() in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(f))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and frames > 0:
                duration = frames / fps
                print(f"   Video: {f.name}  |  {duration:.3f}s @ {fps:.0f}fps")
                return duration
    raise FileNotFoundError(f"No video file found in {folder}")


def load_imu(folder: Path) -> pd.DataFrame:
    """
    Load WatchAccelerometer.csv + WatchGyroscope.csv,
    merge on nearest timestamp, return unified DataFrame.
    Columns: seconds_elapsed, ax, ay, az, gx, gy, gz
    """
    accel_path = folder / ACCEL_FILE
    gyro_path  = folder / GYRO_FILE

    if not accel_path.exists():
        raise FileNotFoundError(f"Missing: {accel_path}")
    if not gyro_path.exists():
        raise FileNotFoundError(f"Missing: {gyro_path}")

    # Note: Sensor Logger exports columns as: time, seconds_elapsed, z, y, x
    accel = pd.read_csv(accel_path)
    gyro  = pd.read_csv(gyro_path)

    # Rename to meaningful names (columns are z, y, x in the CSVs)
    accel = accel.rename(columns={"x": "ax", "y": "ay", "z": "az"})
    gyro  = gyro.rename(columns={"x": "gx", "y": "gy", "z": "gz"})

    # Sort by time
    accel = accel.sort_values("time").reset_index(drop=True)
    gyro  = gyro.sort_values("time").reset_index(drop=True)

    # Merge on nearest timestamp (merge_asof needs sorted keys)
    merged = pd.merge_asof(
        accel[["time", "seconds_elapsed", "ax", "ay", "az"]],
        gyro[["time", "gx", "gy", "gz"]],
        on="time",
        tolerance=50_000_000,   # 50ms in nanoseconds
        direction="nearest"
    ).dropna()

    # Estimate sampling rate from accel timestamps
    dt_ns = accel["time"].diff().median()
    fs = 1e9 / dt_ns  # Hz
    print(f"  📡 IMU: {len(merged)} samples  |  {merged['seconds_elapsed'].iloc[-1]:.2f}s  |  {fs:.1f} Hz")
    return merged, fs


def find_sprint_window(imu: pd.DataFrame, video_dur: float, fs: float) -> tuple:
    """
    Sliding window search: find the window of length video_dur
    with the highest mean acceleration magnitude.

    Returns (start_idx, end_idx).
    """
    # Compute accel magnitude
    mag = np.sqrt(imu["ax"]**2 + imu["ay"]**2 + imu["az"]**2)

    # Smooth to remove single-sample spikes
    mag_smooth = mag.rolling(window=5, center=True).mean().fillna(mag)

    # Window size in samples
    win_samples = int(round(video_dur * fs))
    total = len(mag_smooth)

    if win_samples >= total:
        # IMU is not longer than video — use all of it
        print("  IMU shorter than or equal to video duration. Using full signal.")
        return 0, total - 1

    # Compute score for each window position
    scores = np.array([
        mag_smooth.iloc[i : i + win_samples].mean()
        for i in range(total - win_samples + 1)
    ])

    best_start = int(np.argmax(scores))
    best_end   = best_start + win_samples - 1

    print(f"   Best window: sample {best_start}→{best_end}  "
          f"({imu['seconds_elapsed'].iloc[best_start]:.3f}s → "
          f"{imu['seconds_elapsed'].iloc[best_end]:.3f}s)")
    return best_start, best_end


def save_trimmed(imu: pd.DataFrame, start: int, end: int,
                 athlete_name: str, output_dir: Path) -> Path:
    """Save trimmed IMU to CSV in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trimmed = imu.iloc[start : end + 1].copy()
    # Reset seconds_elapsed to start at 0
    trimmed["seconds_elapsed"] = (
        trimmed["seconds_elapsed"] - trimmed["seconds_elapsed"].iloc[0]
    )
    out_path = output_dir / f"{athlete_name}_synced_imu.csv"
    trimmed.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(trimmed)} samples)")
    return out_path


def plot_sync(imu: pd.DataFrame, start: int, end: int,
              athlete_name: str, output_dir: Path):
    """Save a visual verification plot."""
    mag = np.sqrt(imu["ax"]**2 + imu["ay"]**2 + imu["az"]**2)
    t   = imu["seconds_elapsed"].values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, mag, color="lightgrey", linewidth=1.2, label="Full IMU signal")
    ax.axvspan(t[start], t[end], color="red", alpha=0.25, label="Selected sprint window")
    ax.plot(t[start:end+1], mag.iloc[start:end+1],
            color="red", linewidth=1.5, label="Trimmed segment")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accel magnitude (m/s²)")
    ax.set_title(f"{athlete_name} — IMU Sync Check\n"
                 f"Window: {t[start]:.3f}s → {t[end]:.3f}s  "
                 f"({t[end]-t[start]:.3f}s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    img_path = output_dir / f"{athlete_name}_sync_check.png"
    fig.savefig(img_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot saved: {img_path}")


# ── MAIN PROCESSING ───────────────────────────────────────────────────────────

def process_athlete(folder: Path):
    athlete_name = folder.name
    print(f"\n{'='*55}")
    print(f"  Athlete: {athlete_name}")
    print(f"{'='*55}")

    try:
        video_dur       = get_video_duration(folder)
        imu, fs         = load_imu(folder)
        start, end      = find_sprint_window(imu, video_dur, fs)
        save_trimmed(imu, start, end, athlete_name, OUTPUT_DIR)
        plot_sync(imu, start, end, athlete_name, OUTPUT_DIR)
        print(f"   Done: {athlete_name}")
    except FileNotFoundError as e:
        print(f"   Skipped ({e})")
    except Exception as e:
        print(f"   Error: {e}")
        raise


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim IMU to match video duration")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", type=Path, help="Single athlete folder")
    group.add_argument("--batch",  type=Path, help="Root folder containing all athlete folders")
    args = parser.parse_args()

    if args.folder:
        process_athlete(args.folder)
    else:
        # Batch: iterate over all subdirectories
        folders = sorted([f for f in args.batch.iterdir() if f.is_dir()])
        print(f"Found {len(folders)} athlete folders in {args.batch}")
        for folder in folders:
            process_athlete(folder)

    print(f"\n\n All done! Results saved to: {OUTPUT_DIR}")
