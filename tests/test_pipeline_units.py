"""Unit tests: runner tracking, velocity labels, features, IMU loading, data-quality gate."""
import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_imu, synthetic_pose
from sprintml import features as F
from sprintml import labels, quality
from sprintml.imu import load_imu
from sprintml.pose import runner_track


def test_runner_track_drops_static_false_detection(pose_df):
    kept = runner_track(pose_df)
    assert kept.frame.min() >= 30                      # the 30 false-detection frames are gone
    assert np.ptp((kept.x23 + kept.x24) / 2) > 200     # and the moving runner is what remains


@pytest.mark.parametrize("speed", [3.5, 4.5, 5.5])
def test_velocity_recovers_ground_truth(params, speed):
    pose = synthetic_pose(speed_statures_per_s=speed)
    v = labels.measure_velocity(pose, params)
    assert v["v_statures_per_s"] == pytest.approx(speed, rel=0.03)
    assert v["fit_r2"] > 0.999
    assert v["v_mps"] == pytest.approx(speed * params["labels"]["assumed_stature_m"], rel=0.03)


def test_velocity_is_camera_distance_invariant(params):
    near = labels.measure_velocity(synthetic_pose(4.5, stature_px=300), params)["v_statures_per_s"]
    far = labels.measure_velocity(synthetic_pose(4.5, stature_px=120), params)["v_statures_per_s"]
    assert near == pytest.approx(far, rel=0.03)


def test_velocity_rejects_untracked_clip(params):
    pose = synthetic_pose()
    pose["detected"] = False
    with pytest.raises(Exception):
        labels.measure_velocity(pose, params)


def test_tiers_are_tertiles():
    v = np.array([4.0, 4.1, 4.2, 4.5, 4.6, 4.7, 5.0, 5.1, 5.2])
    thr = labels.tier_thresholds(v)
    t = labels.assign_tiers(v, thr)
    assert np.bincount(t).tolist() == [3, 3, 3] and t[0] == 0 and t[-1] == 2


def test_features_complete_and_physical(params, pose_df):
    a, g = synthetic_imu()
    x = F.extract_features(pose_df, load_imu(a, g), params)
    assert set(x["features"]) == set(F.FEATURES) and len(F.FEATURES) == 19
    assert all(np.isfinite(v) for v in x["features"].values())
    assert 60 < x["features"]["knee_angle_mean"] < 180
    assert 95 < x["meta"]["imu_hz"] < 105


def test_features_deterministic(params, pose_df):
    a, g = synthetic_imu()
    x1 = F.extract_features(pose_df, load_imu(a, g), params)["features"]
    x2 = F.extract_features(pose_df.copy(), load_imu(a.copy(), g.copy()), params)["features"]
    assert x1 == x2


def test_imu_loader_validation():
    a, g = synthetic_imu()
    with pytest.raises(ValueError):
        load_imu(a.drop(columns=["x"]), g)
    m = load_imu(a, g)
    assert {"x_a", "x_g", "t"} <= set(m.columns) and len(m) == len(a)
    # accepts seconds_elapsed-only exports
    m2 = load_imu(a.drop(columns=["time"]), g.drop(columns=["time"]))
    assert len(m2) == len(a)


# ── data-quality gate ────────────────────────────────────────────────────────
def good():
    feats = {k: 1.0 for k in F.FEATURES}
    meta = {"fps": 59.8, "imu_hz": 99.4, "fit_r2": 0.999, "frames_used": 80, "wrist_cadence_hz": 2.0,
            "gyro_cadence_hz": 2.1}
    return feats, meta


def test_quality_passes_good_recording(params):
    assert quality.check_recording(*good(), params)["passed"]


@pytest.mark.parametrize("mutate,failed", [
    (lambda f, m: m.update(fps=30.0), "video_fps"),
    (lambda f, m: m.update(imu_hz=50.0), "imu_rate_hz"),
    (lambda f, m: m.update(fit_r2=0.5), "runner_track_r2"),
    (lambda f, m: m.update(frames_used=10), "tracked_frames"),
    (lambda f, m: f.update(gct_ms=float("nan")), "features_finite"),
    (lambda f, m: m.update(gyro_cadence_hz=3.0), "video_imu_cadence_agreement_hz"),
])
def test_quality_rule_triggers(params, mutate, failed):
    f, m = good()
    mutate(f, m)
    q = quality.check_recording(f, m, params)
    assert not q["passed"] and failed in q["failed"]


def test_clock_check_quarantines_mismatched_watch_file(params):
    base = "2026-03-25T12:00:{:02d}+00:00"
    recs = []
    for i, offset in enumerate([2.0, 2.4, 1.9, 2.2, -400.0]):    # last: watch file from another run
        f, m = good()
        m["imu_end_utc"] = base.format(10)
        m["video_creation_utc"] = pd.Timestamp("2026-03-25T12:00:10+00:00").__add__(
            pd.Timedelta(seconds=offset)).isoformat()
        recs.append({"clip": f"r{i}", "meta": m, "quality": quality.check_recording(f, m, params)})
    quality.apply_clock_check(recs, params)
    assert [r["quality"]["passed"] for r in recs] == [True, True, True, True, False]
    assert "clock_offset_consistency_s" in recs[-1]["quality"]["failed"]
