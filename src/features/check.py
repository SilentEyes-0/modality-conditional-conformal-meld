"""Sanity-check an assembled feature file BEFORE using it.
  python -m src.features.check --config configs/experiment.yaml --split train
Reports shapes, NaN/inf, failed items (ok=False), all-zero rows, per-modality statistics and CSV alignment."""
import argparse, os
import numpy as np
from ..constants import DIMS, MODALITIES


def check_features(d, expected_ids=None):
    rep = {"n": int(len(d["ids"])), "problems": []}
    for k, m in enumerate(MODALITIES):
        X = d["feats"][m]; ok = d["ok"][:, k]
        rep[m] = {"dim": int(X.shape[1]), "nan": int(np.isnan(X).sum()), "inf": int(np.isinf(X).sum()), "failed": int((~ok).sum()),
                  "all_zero_rows_among_ok": int(((np.abs(X).sum(1) == 0) & ok).sum()), "mean": float(X[ok].mean()) if ok.any() else None,
                  "std": float(X[ok].std()) if ok.any() else None}
        if X.shape[1] != DIMS[m]: rep["problems"].append(f"{m}: dimension {X.shape[1]} != {DIMS[m]}")
        if rep[m]["nan"] or rep[m]["inf"]: rep["problems"].append(f"{m}: NaN or inf present")
        if rep[m]["all_zero_rows_among_ok"]: rep["problems"].append(f"{m}: zero vectors flagged as ok")
        if (~ok).any() and np.abs(X[~ok]).sum() != 0: rep["problems"].append(f"{m}: failed rows are not zero-filled")
    rep["failed_ids"] = d["ids"][~d["ok"].all(1)].tolist()
    if len(set(d["ids"].tolist())) != len(d["ids"]): rep["problems"].append("duplicate ids")
    if expected_ids is not None and list(d["ids"]) != list(expected_ids): rep["problems"].append("ids differ from CSV order")
    return rep


if __name__ == "__main__":
    import json
    from ..utils import load_config
    from ..data.meld import load_split_csv
    from .store import load_features
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/experiment.yaml"); ap.add_argument("--split", required=True)
    ap.add_argument("--condition"); a = ap.parse_args(); cfg = load_config(a.config)
    name = f"{a.split}__{a.condition}.npz" if a.condition else f"{a.split}.npz"
    df = load_split_csv(os.path.join(cfg["data"]["root"], cfg["data"]["csv"][a.split]))
    print(json.dumps(check_features(load_features(os.path.join(cfg["paths"]["features"], name)), df["id"].tolist()), indent=2))
