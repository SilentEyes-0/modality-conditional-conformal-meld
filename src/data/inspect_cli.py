"""P1 command: report counts, class counts and missing media for each MELD split.
  python -m src.data.inspect_cli --config configs/experiment.yaml
Compare 'counts_match_official' with the official numbers; explain every difference before going on."""
import argparse, json
from ..utils import load_config
from .meld import inspect_split

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/experiment.yaml"); a = ap.parse_args()
    cfg = load_config(a.config)
    print(json.dumps({s: inspect_split(cfg, s) for s in ["train", "dev", "test"]}, indent=2))
