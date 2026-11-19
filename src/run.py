"""Command line: inspect | freeze | run | resplits | aggregate.
  python -m src.run run --config configs/experiment.yaml --mode dev     # development (dev 'select' subset)
  python -m src.run freeze --config configs/experiment.yaml             # lock analysis_plan before test
  python -m src.run run --config configs/experiment.yaml --mode final   # test set, requires lock
"""
import argparse, json
from .utils import load_config, ensure_dir
from . import pipeline
from .evaluation.aggregate import aggregate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "freeze", "run", "resplits", "aggregate"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--mode", choices=["dev", "final"], default="dev")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--k", type=int, default=20, help="number of resplits")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config); out = ensure_dir(cfg["paths"]["outputs"])
    if a.cmd == "inspect":
        from .data.meld import inspect_split
        rep = {s: inspect_split(cfg, s) for s in ["train", "dev", "test"]}
        print(json.dumps(rep, indent=2)); return rep
    if a.cmd == "freeze":
        print("frozen:", pipeline.freeze_plan(cfg)); return
    seeds = a.seeds or cfg["train"]["seeds"]
    if a.cmd == "run":
        split = "test" if a.mode == "final" else "val"
        lock = None
        if a.mode == "final": lockpath, lock = pipeline.check_lock_for_test(cfg, a.force)
        for sd in seeds:
            pipeline.run_experiment(cfg, sd, cfg["split"]["split_seed"], split, out); print("done seed", sd)
        aggregate(out, split)
        if lock is not None: pipeline.mark_test_evaluated(lockpath, lock)
    if a.cmd == "resplits":      # development-only variability of coverage across calibration draws
        for i in range(a.k):
            pipeline.run_experiment(cfg, seeds[0], cfg["split"]["split_seed"] + i, "val", ensure_dir(f"{out}/resplits"))
        aggregate(f"{out}/resplits", "val")
    if a.cmd == "aggregate":
        aggregate(out, "test" if a.mode == "final" else "val")


if __name__ == "__main__":
    main()
