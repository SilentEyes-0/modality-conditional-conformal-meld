import numpy as np
from .test_p5_pipeline import cfg  # noqa: F401  (reuse the synthetic-data fixture)
from src.evaluation.shift_check import shift_report


def test_shift_report_structure_and_control(cfg):
    cfg["train"].update(epochs=4, patience=3)
    r = shift_report(cfg, seed=0, n_control=5)
    assert list(r["pattern"]) == ["111", "110", "101", "011", "100", "010", "001"]
    assert ((r["ks_cal_vs_dev"] >= 0) & (r["ks_cal_vs_dev"] <= 1)).all()
    assert (r["ks_control_max"] >= r["ks_control_mean"] - 1e-12).all() and (r["n_cal"] > 0).all() and (r["n_dev"] > 0).all()
