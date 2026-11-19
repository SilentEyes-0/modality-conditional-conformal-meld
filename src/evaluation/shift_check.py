"""Development diagnostic: do the calibration scores (held-out TRAIN dialogues) look like the evaluation scores (DEV)?
Conformal coverage assumes they are exchangeable. For each pattern this compares nonconformity-score distributions of the
calibration set with the dev temperature-fit subset (never used for training or early stopping), against a size-matched control
(a random calibration subset with as many dialogues as the dev subset versus the rest of the calibration set). A cal-vs-dev KS statistic far above the control suggests a real shift.
KS p-values are NOT reported: utterances within a dialogue are dependent, so naive p-values would be too optimistic.
  python -m src.evaluation.shift_check --config configs/experiment.yaml --seed 42"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from ..constants import PATTERNS, pattern_name
from ..features.store import load_features
from ..data.splits import make_partitions, split_dialogues, indices_for
from ..masking import Standardizer
from ..models.train import train_fusion
from ..conformal.core import scores_from_probs
from ..models.mlp import softmax
from ..pipeline import drop_failed, _std_set, _logits_by_mask
from ..features.store import subset


def shift_report(cfg, seed=42, split_seed=None, n_control=20):
    split_seed = cfg["split"]["split_seed"] if split_seed is None else split_seed
    fd = cfg["paths"]["features"]
    tr, _ = drop_failed(load_features(f"{fd}/train.npz")); dv, _ = drop_failed(load_features(f"{fd}/dev.npz"))
    parts = make_partitions(tr["dialogue"], dv["dialogue"], cfg["split"]["cal_frac"], split_seed)
    raw = {k: subset(tr if k in ("fit", "cal") else dv, parts[k]) for k in ["fit", "cal", "select", "temp"]}
    std = Standardizer().fit(raw["fit"]["feats"]); S = {k: _std_set(v, std) for k, v in raw.items()}
    model, _ = train_fusion(S["fit"], S["select"], "dropout", cfg["train"], seed)
    lc, ld = _logits_by_mask(model, S["cal"]["feats"]), _logits_by_mask(model, S["temp"]["feats"])
    rng = np.random.default_rng(seed); rows = []
    for m in PATTERNS:
        sc = scores_from_probs(softmax(lc[m][0]), S["cal"]["emotion"]); sd = scores_from_probs(softmax(ld[m][0]), S["temp"]["emotion"])
        ctrl = []
        frac = len(set(S["temp"]["dialogue"].tolist())) / len(set(S["cal"]["dialogue"].tolist()))   # size-matched control
        for _ in range(n_control):                       # small side has as many dialogues as the dev subset
            big, small = split_dialogues(S["cal"]["dialogue"], frac, int(rng.integers(1_000_000)))
            ib, is_ = indices_for(S["cal"]["dialogue"], big), indices_for(S["cal"]["dialogue"], small)
            ctrl.append(ks_2samp(sc[ib], sc[is_]).statistic)
        rows.append({"pattern": pattern_name(m), "n_cal": len(sc), "n_dev": len(sd), "mean_score_cal": sc.mean(), "mean_score_dev": sd.mean(),
                     "ks_cal_vs_dev": ks_2samp(sc, sd).statistic, "ks_control_mean": float(np.mean(ctrl)), "ks_control_max": float(np.max(ctrl))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from ..utils import load_config
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/experiment.yaml"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(shift_report(load_config(a.config), a.seed).round(3).to_string(index=False))
