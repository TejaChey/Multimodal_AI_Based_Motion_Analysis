"""python -m sprintml.cli {pose,features,validate,train,retrain,monitor,loop}"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from . import pipeline
from .config import load_params


def main():
    ap = argparse.ArgumentParser(prog="sprintml")
    ap.add_argument("cmd", choices=["pose", "features", "validate", "train", "retrain", "monitor", "loop", "replay"])
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    a = ap.parse_args()
    p = load_params()
    if a.cmd == "pose":
        pipeline.stage_pose(p)
    elif a.cmd == "features":
        pipeline.stage_features(p)
    elif a.cmd == "validate":
        pipeline.stage_validate(p)
    elif a.cmd == "train":
        pipeline.stage_train(p, n_perm=a.n_perm)
    elif a.cmd == "retrain":
        print(json.dumps(pipeline.maybe_retrain(p, force=a.force, n_perm=a.n_perm), indent=2, default=float))
    elif a.cmd == "monitor":
        r = pipeline.run_monitor(p)
        print(json.dumps({k: r.get(k) for k in ("status", "armed", "n_reference", "n_window", "drifted_features")}))
    elif a.cmd == "replay":
        from .replay import run
        r = run(a.n_perm or 100)
        print(json.dumps({k: v for k, v in r.items() if k != "steps"}))
    elif a.cmd == "loop":
        pipeline.loop(p, a.interval)


if __name__ == "__main__":
    main()
