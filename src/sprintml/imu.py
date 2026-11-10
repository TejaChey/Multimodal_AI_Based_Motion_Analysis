"""Smartwatch IMU loading (Sensor Logger CSV: time[ns], seconds_elapsed, x, y, z)."""
import numpy as np
import pandas as pd


def _read(src) -> pd.DataFrame:
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ("x", "y", "z"):
        if c not in df.columns:
            raise ValueError(f"IMU file missing column '{c}' (found {list(df.columns)})")
    if "time" not in df.columns:
        if "seconds_elapsed" not in df.columns:
            raise ValueError("IMU file needs a 'time' (ns) or 'seconds_elapsed' column")
        df["time"] = (df["seconds_elapsed"] * 1e9).astype("int64")
    return df.sort_values("time").reset_index(drop=True)


def load_imu(accel, gyro) -> pd.DataFrame:
    """Merge accelerometer and gyroscope by nearest timestamp (20 ms tolerance)."""
    a, g = _read(accel), _read(gyro)
    m = pd.merge_asof(a[["time", "x", "y", "z"]], g[["time", "x", "y", "z"]], on="time",
                      suffixes=("_a", "_g"), tolerance=20_000_000, direction="nearest").dropna()
    if len(m) < 20:
        raise ValueError("fewer than 20 synchronised IMU samples")
    m = m.reset_index(drop=True)
    m["t"] = (m.time - m.time.iloc[0]) / 1e9
    return m
