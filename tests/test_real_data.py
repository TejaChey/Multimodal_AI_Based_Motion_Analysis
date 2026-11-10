"""Regression test against the validated paper numbers (needs the raw recordings; skipped in CI)."""
import numpy as np
import pandas as pd
import pytest

from sprintml import config, features as F
from sprintml.imu import load_imu

GOLD = config.ROOT / "paper_revision" / "results"
pytestmark = pytest.mark.real_data


@pytest.mark.skipif(not (GOLD / "features.csv").exists() or not config.raw_dir().exists(),
                    reason="raw recordings / validated results not available")
@pytest.mark.parametrize("clip", ["Amith", "Tejas", "Yeshwanth_trail_2", "Mohammed_araar"])
def test_package_reproduces_validated_features(params, clip):
    gold = pd.read_csv(GOLD / "features.csv").set_index("clip")
    pose = pd.read_csv(GOLD / "pose" / f"{clip}.csv")
    d = config.raw_dir() / clip
    x = F.extract_features(pose, load_imu(d / "WatchAccelerometer.csv", d / "WatchGyroscope.csv"), params)
    for k, v in x["features"].items():
        assert v == pytest.approx(gold.loc[clip, k], rel=1e-6, abs=1e-6), k
    assert x["meta"]["v_mps"] == pytest.approx(gold.loc[clip, "v_approx_mps"], rel=1e-6)
