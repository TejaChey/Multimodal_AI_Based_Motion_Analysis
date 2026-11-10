"""Paths and parameters. Everything tunable lives in params.yaml."""
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = ["Beginner", "Intermediate", "Elite"]
MODEL_NAME = "sprint-tier-classifier"


def art_dir() -> Path:
    """Artifact root (re-read each call so tests/replays can redirect it)."""
    p = Path(os.environ.get("SPRINTML_ARTIFACTS", ROOT / "artifacts"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir() -> Path:
    return Path(os.environ.get("SPRINTML_RAW", ROOT / "Phase_2" / "data" / "sprint_raw"))


def load_params(path=None) -> dict:
    path = Path(path or os.environ.get("SPRINTML_PARAMS", ROOT / "params.yaml"))
    return yaml.safe_load(path.read_text())


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def dvc_data_hash() -> str:
    """md5 of the raw-data directory recorded by DVC (data lineage), if available."""
    lock = ROOT / "dvc.lock"
    if lock.exists():
        try:
            d = yaml.safe_load(lock.read_text())
            for dep in d["stages"]["pose"]["deps"]:
                if "sprint_raw" in dep["path"]:
                    return dep["md5"]
        except Exception:
            pass
    return "untracked"
