#!/usr/bin/env python3
"""
Phase 2 IMU Feature Extractor
-----------------------------
Accepts separate Galaxy Watch accelerometer & gyroscope CSV files,
merges them, and extracts 11 sprint-specific IMU features.
"""

import numpy as np
import pandas as pd
from scipy.signal import welch
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent

INPUT_DIR  = PROJECT_ROOT / "Phase_2" / "results" / "synced_imu"
OUTPUT_DIR = PROJECT_ROOT / "Phase_2" / "results" / "imu_features"
OUTPUT_CSV = OUTPUT_DIR / "imu_features.csv"

# ── MATH & FEATURES ───────────────────────────────────────────────────────────

def get_magnitude(x, y, z):
    """Euclidean norm of 3D array"""
    return np.sqrt(x**2 + y**2 + z**2)

def get_dominant_frequency(signal, fs):
    """Finds peak frequency using Welch's power spectral density"""
    if len(signal) < 10:
        return 0.0
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= 1.0) & (freqs <= 5.0)
    if not mask.any():
        return float(freqs[np.argmax(psd)])
    valid_freqs = freqs[mask]
    valid_psd = psd[mask]
    return float(valid_freqs[np.argmax(valid_psd)])

def get_sparc_smoothness(signal, fs):
    """
    Computes spectral arc length (SPARC) as a measure of smoothness.
    Higher (closer to 0) = smoother, Lower (more negative) = jerky.
    """
    if len(signal) < 10:
        return 0.0
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    psd_norm = psd / (psd.max() + 1e-10)
    freq_spacing = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    arc_length = -np.sum(np.sqrt(freq_spacing**2 + np.diff(psd_norm)**2))
    return float(arc_length)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _normalise_columns(df, axis_map):
    """Rename columns to standard names using a case-insensitive alias lookup."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for standard, aliases in axis_map.items():
        for alias in aliases:
            if alias in cols_lower:
                rename[cols_lower[alias]] = standard
                break
    return df.rename(columns=rename)

def _smart_read_csv(path: Path) -> pd.DataFrame:
    """
    Robustly read Galaxy Watch CSVs.
    Handles: MultiIndex headers, unit rows, various column name formats.
    """
    df = pd.read_csv(path, header=0)

    # Flatten MultiIndex (two header rows in some exports)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            ' '.join(str(c).strip() for c in col
                     if str(c).strip() not in ('', 'nan'))
            for col in df.columns
        ]
    else:
        df.columns = [str(c).strip() for c in df.columns]

    # Drop first data row if it looks like units (non-numeric text)
    if len(df) > 0:
        first = df.iloc[0]
        try:
            pd.to_numeric(first.iloc[0])  # try converting the first cell
        except (ValueError, TypeError):
            # First row is a units row — skip it and re-cast columns
            df = df.iloc[1:].reset_index(drop=True)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ── MAIN PROCESSOR ────────────────────────────────────────────────────────────

def process_file(accel_path: Path, gyro_path: Path) -> dict:
    """
    Read separate Galaxy Watch accelerometer & gyroscope CSVs,
    merge them by index alignment, and return 11 aggregated sprint IMU features.
    """
    df_a = _smart_read_csv(accel_path)
    df_g = _smart_read_csv(gyro_path)

    accel_cols_found = list(df_a.columns)
    gyro_cols_found  = list(df_g.columns)

    if len(df_a) < 10 or len(df_g) < 10:
        raise ValueError("Too few IMU samples to process.")

    # ── Normalise column names ────────────────────────────────────────────────
    accel_map = {
        'ax': ['ax', 'accel x', 'accelerometeraccx', 'acceleration x',
               'linear acceleration x (m/s^2)', 'linear acceleration x',
               'x(m/s^2)', 'linaccx', 'acc_x', 'x'],
        'ay': ['ay', 'accel y', 'accelerometeraccy', 'acceleration y',
               'linear acceleration y (m/s^2)', 'linear acceleration y',
               'y(m/s^2)', 'linaccy', 'acc_y', 'y'],
        'az': ['az', 'accel z', 'accelerometeraccz', 'acceleration z',
               'linear acceleration z (m/s^2)', 'linear acceleration z',
               'z(m/s^2)', 'linaccz', 'acc_z', 'z'],
        'seconds_elapsed': ['seconds_elapsed', 'time (s)', 'time(s)',
                            'time', 'timestamp', 'elapsed time',
                            'seconds', 'time_s', 'time(ns)'],
    }
    gyro_map = {
        'gx': ['gx', 'gyro x', 'gyroscopex', 'angular velocity x',
               'angular velocity x (rad/s)', 'wx', 'gyro_x', 'x'],
        'gy': ['gy', 'gyro y', 'gyroscopey', 'angular velocity y',
               'angular velocity y (rad/s)', 'wy', 'gyro_y', 'y'],
        'gz': ['gz', 'gyro z', 'gyroscopez', 'angular velocity z',
               'angular velocity z (rad/s)', 'wz', 'gyro_z', 'z'],
        'seconds_elapsed': ['seconds_elapsed', 'time (s)', 'time(s)',
                            'time', 'timestamp', 'elapsed time',
                            'seconds', 'time_s', 'time(ns)'],
    }

    try:
        df_a = _normalise_columns(df_a, accel_map)
        df_g = _normalise_columns(df_g, gyro_map)
    except Exception as e:
        raise ValueError(
            f"Column rename failed: {e}. "
            f"Accel columns: {accel_cols_found}. "
            f"Gyro columns: {gyro_cols_found}."
        )

    # Validate required columns — show actual names on failure
    for col in ['ax', 'ay', 'az']:
        if col not in df_a.columns:
            raise KeyError(
                f"Accel CSV: cannot find '{col}'. "
                f"Detected columns: {accel_cols_found}. "
                f"Please share these column names so we can add mapping."
            )
    for col in ['gx', 'gy', 'gz']:
        if col not in df_g.columns:
            raise KeyError(
                f"Gyro CSV: cannot find '{col}'. "
                f"Detected columns: {gyro_cols_found}. "
                f"Please share these column names so we can add mapping."
            )

    # ── Merge by index alignment (trim to shorter length) ────────────────────
    min_len = min(len(df_a), len(df_g))
    df_a = df_a.iloc[:min_len].reset_index(drop=True)
    df_g = df_g.iloc[:min_len].reset_index(drop=True)

    df = pd.DataFrame({
        'ax': pd.to_numeric(df_a['ax'], errors='coerce'),
        'ay': pd.to_numeric(df_a['ay'], errors='coerce'),
        'az': pd.to_numeric(df_a['az'], errors='coerce'),
        'gx': pd.to_numeric(df_g['gx'], errors='coerce'),
        'gy': pd.to_numeric(df_g['gy'], errors='coerce'),
        'gz': pd.to_numeric(df_g['gz'], errors='coerce'),
    }).dropna().reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("Too few valid numeric IMU rows after parsing.")

    # Reconstruct time axis
    if 'seconds_elapsed' in df_a.columns:
        t = pd.to_numeric(df_a['seconds_elapsed'], errors='coerce').dropna()
        df['seconds_elapsed'] = t.values[:len(df)]
    elif 'seconds_elapsed' in df_g.columns:
        t = pd.to_numeric(df_g['seconds_elapsed'], errors='coerce').dropna()
        df['seconds_elapsed'] = t.values[:len(df)]
    else:
        df['seconds_elapsed'] = np.arange(len(df)) / 38.0

    # Handle nanosecond timestamps (convert to seconds)
    if df['seconds_elapsed'].max() > 1e9:
        df['seconds_elapsed'] = df['seconds_elapsed'] / 1e9

    # ── Feature extraction ────────────────────────────────────────────────────
    duration = float(df['seconds_elapsed'].iloc[-1]) - float(df['seconds_elapsed'].iloc[0])
    fs = (len(df) - 1) / duration if duration > 0 else 38.0

    accel_mag = get_magnitude(df['ax'], df['ay'], df['az'])
    gyro_mag  = get_magnitude(df['gx'], df['gy'], df['gz'])

    a_freq = get_dominant_frequency(accel_mag, fs)
    g_freq = get_dominant_frequency(gyro_mag, fs)

    return {
        'athlete':                    accel_path.stem,
        'peak_accel_mag':             round(float(accel_mag.max()), 3),
        'mean_accel_mag':             round(float(accel_mag.mean()), 3),
        'std_accel_mag':              round(float(accel_mag.std()), 3),
        'stride_rate_accel_hz':       round(a_freq, 2),
        'movement_smoothness_accel':  round(get_sparc_smoothness(accel_mag, fs), 3),
        'peak_gyro_mag':              round(float(gyro_mag.max()), 3),
        'mean_gyro_mag':              round(float(gyro_mag.mean()), 3),
        'std_gyro_mag':               round(float(gyro_mag.std()), 3),
        'stride_rate_gyro_hz':        round(g_freq, 2),
        'movement_smoothness_gyro':   round(get_sparc_smoothness(gyro_mag, fs), 3),
        'steps_per_min_gyro':         round(g_freq * 60, 1),
    }


def main():
    print(f"{'='*50}\n  Phase 2: Sprint IMU Feature Extractor\n{'='*50}")
    print("Note: This script now expects separate accel & gyro CSV pairs.")


if __name__ == "__main__":
    main()
