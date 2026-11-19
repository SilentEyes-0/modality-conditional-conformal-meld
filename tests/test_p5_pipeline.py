import copy, os, numpy as np, pandas as pd, pytest
from src.utils import load_config
from src.data.synthetic import write_synthetic_dataset, corrupt_modality
from src import pipeline
from src.run import main
from src.evaluation import metrics as M

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")

def test_metrics_basic():
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]); y = np.array([0, 1, 1])
    r = M.classification_metrics(p, y)
    assert abs(r["acc"] - 2 / 3) < 1e-9 and 0 <= r["ece"] <= 1 and r["brier"] > 0
    cov, sz = M.set_stats(np.array([[1, 0], [1, 1], [1, 0]], bool), y)
    assert cov.tolist() == [1, 1, 0] and sz.tolist() == [1, 2, 1]


def test_cluster_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0); dia = np.repeat(np.arange(50), 6); v = (rng.random(300) < 0.9).astype(float)
    lo, hi = M.cluster_bootstrap(v, dia, 400, rng); assert lo[0] <= v.mean() <= hi[0]


@pytest.fixture()
def cfg(tmp_path):
    c = load_config(CFG)
    feats = str(tmp_path / "feats"); write_synthetic_dataset(feats, seed=0, sizes=(120, 20, 40))
    for sp in ["train", "dev", "test"]: corrupt_modality(feats, sp, "audio", "audio_noise", 3.0)
    c["paths"] = {"features": feats, "outputs": str(tmp_path / "out")}
    c["train"].update(epochs=4, patience=3, seeds=[0, 1], h1=32, h2=16)
    c["analysis_plan"].update(bootstrap_B=30, alphas=[0.10], n_min_list=[30], cal_profiles=["uniform", "skewed"],
                              corruption=[{"name": "audio_noise", "modality": "audio", "calibration": "matched"},
                                          {"name": "audio_noise", "modality": "audio", "calibration": "mismatched"}])
    return c


def _cfg_path(cfg, tmp_path):
    import yaml; p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(cfg)); return str(p)


def test_dev_run_outputs_and_invariants(cfg):
    conf, cls, meta = pipeline.run_experiment(cfg, 0, 42, "val")
    df = pd.DataFrame(conf)
    assert {"global_111", "global_pooled", "conditional"} <= set(df["method"].dropna())
    pat = df[df["pattern"].str.fullmatch(r"[01]{3}")]
    assert (pat["coverage"].between(0, 1)).all() and (pat["size"] >= 0).all()
    assert (pat["cov_lo"] <= pat["coverage"] + 1e-9).all() and (pat["cov_hi"] >= pat["coverage"] - 1e-9).all()
    # Design A: conditional uses level 0 everywhere and the same n for every pattern
    a = pat[(pat.design == "A") & (pat.method == "conditional") & (pat.condition == "clean")]
    assert (a["level"].astype(str) == "0").all() and a["n_cal_group"].nunique() == 1
    # global-111 threshold equals conditional threshold for pattern 111 in Design A -> identical sets on pattern 111
    g = pat[(pat.design == "A") & (pat.pattern == "111") & (pat.condition == "clean") & (pat.variant == "R") & (pat.task == "emotion")]
    cg = g[g.method == "global_111"].iloc[0]; cc = g[g.method == "conditional"].iloc[0]
    assert abs(cg.coverage - cc.coverage) < 1e-12 and abs(cg["size"] - cc["size"]) < 1e-12
    # corruption rows exist for both calibration regimes
    assert {"audio_noise|matched", "audio_noise|mismatched"} <= set(df["condition"])
    assert meta["temperatures"]["emotion"] > 0 and "plan_hash" in meta
    # S3 rows contain both raw and temperature-scaled calibration metrics
    c = pd.DataFrame(cls); assert {"ece_R", "ece_T", "macro_f1_R"} <= set(c.columns)
    assert {"S1-text", "S1-audio", "S1-video", "S2", "S3"} <= set(c["system"])


def test_determinism(cfg):
    a = pd.DataFrame(pipeline.run_experiment(cfg, 0, 42, "val")[0]); b = pd.DataFrame(pipeline.run_experiment(cfg, 0, 42, "val")[0])
    pd.testing.assert_frame_equal(a, b)


def test_conditional_beats_global_on_pattern_coverage_in_design_A(cfg):
    """Structural sanity on SYNTHETIC data only: global-111 should under-cover weaker patterns more than conditional."""
    cfg["analysis_plan"]["corruption"] = []; cfg["analysis_plan"]["designs"] = ["A"]
    cfg["train"].update(epochs=15, patience=6, h1=64, h2=32)
    df = pd.DataFrame(pipeline.run_experiment(cfg, 0, 42, "val")[0])
    pat = df[df["pattern"].str.fullmatch(r"[01]{3}") & (df.condition == "clean") & (df.variant == "R") & (df.task == "emotion")]
    dev = lambda m: (pat[pat.method == m]["coverage"] - 0.9).abs().max()
    assert dev("global_111") > dev("conditional") - 1e-9


def test_final_mode_requires_lock_then_runs_once(cfg, tmp_path):
    path = _cfg_path(cfg, tmp_path)
    with pytest.raises(RuntimeError): main(["run", "--config", path, "--mode", "final", "--seeds", "0"])
    main(["freeze", "--config", path])
    main(["run", "--config", path, "--mode", "final", "--seeds", "0"])
    assert os.path.exists(os.path.join(cfg["paths"]["outputs"], "summary_conformal_test.csv"))
    with pytest.raises(RuntimeError): main(["run", "--config", path, "--mode", "final", "--seeds", "0"])   # already evaluated
    changed = copy.deepcopy(cfg); changed["analysis_plan"]["n_min_list"] = [10]
    with pytest.raises(RuntimeError): pipeline.check_lock_for_test(changed, force=True)                    # plan edited after freeze


def test_aggregate_and_worst_pattern(cfg, tmp_path):
    path = _cfg_path(cfg, tmp_path)
    main(["run", "--config", path, "--mode", "dev", "--seeds", "0", "1"])
    out = cfg["paths"]["outputs"]
    s = pd.read_csv(f"{out}/summary_conformal_val.csv"); w = pd.read_csv(f"{out}/worst_pattern_val.csv")
    assert (s["runs"] == 2).all() and {"worst_deviation", "worst_undercoverage"} <= set(w.columns)


def test_taxonomy_ablation_and_calibration_subsampling(cfg):
    cfg["analysis_plan"].update(corruption=[], designs=["B"], taxonomies=["cardinality", "text"], cal_keep_frac=1.0)
    full = pd.DataFrame(pipeline.run_experiment(cfg, 0, 42, "val")[0])
    assert {"conditional_cardinality", "conditional_text"} <= set(full["method"])
    cfg["analysis_plan"]["cal_keep_frac"] = 0.5
    half = pd.DataFrame(pipeline.run_experiment(cfg, 0, 42, "val")[0])
    n_full = full[(full.method == "global_pooled") & (full.pattern == "mixture")]["n_cal_group"].iloc[0]
    n_half = half[(half.method == "global_pooled") & (half.pattern == "mixture")]["n_cal_group"].iloc[0]
    assert n_half < 0.7 * n_full


def test_utterances_with_failed_extraction_are_excluded_everywhere(cfg):
    from src.features.store import load_features, save_features
    p = cfg["paths"]["features"] + "/dev.npz"; d = load_features(p)
    d["ok"][3, 1] = False; d["ok"][3, 2] = False; failed_id = str(d["ids"][3])
    for k in ["audio", "video"]: d["feats"][k][3] = 0
    save_features(p, d["ids"], d["dialogue"], d["feats"], d["emotion"], d["sentiment"], d["ok"])
    cfg["analysis_plan"].update(corruption=[], designs=["A"], taxonomies=[])
    conf, cls, meta = pipeline.run_experiment(cfg, 0, 42, "val")
    assert meta["excluded_failed_extraction"]["dev"] == [failed_id] and meta["excluded_failed_extraction"]["train"] == []
    assert meta["n"]["select"] + meta["n"]["temp"] == len(d["ids"]) - 1


def test_summary_keeps_pattern_names_with_leading_zeros(cfg, tmp_path):
    cfg["analysis_plan"].update(corruption=[], designs=["A"], taxonomies=[])
    path = _cfg_path(cfg, tmp_path)
    main(["run", "--config", path, "--mode", "dev", "--seeds", "0"])
    out = cfg["paths"]["outputs"]
    k = pd.read_csv(f"{out}/summary_classification_val.csv", dtype={"pattern": str})
    assert {"001", "010", "011", "100", "101", "110", "111"} <= set(k["pattern"])
    assert "001" in open(f"{out}/summary_classification_val.csv").read().replace(",", " ").split()


def _mark_failed(path, row):
    from src.features.store import load_features, save_features
    d = load_features(path); d["ok"][row, 1] = False; d["ok"][row, 2] = False
    for k in ["audio", "video"]: d["feats"][k][row] = 0
    save_features(path, d["ids"], d["dialogue"], d["feats"], d["emotion"], d["sentiment"], d["ok"])


def test_corrupted_features_are_aligned_after_exclusions(cfg):
    from src.features.store import load_features, subset
    f = cfg["paths"]["features"]
    for name in ["train", "train__audio_noise", "dev", "dev__audio_noise"]: _mark_failed(f"{f}/{name}.npz", 5)
    clean, _ = pipeline.drop_failed(load_features(f"{f}/train.npz")); cond = load_features(f"{f}/train__audio_noise.npz")
    idx = np.arange(40, 60)
    assert not np.array_equal(subset(clean, idx)["ids"], subset(cond, idx)["ids"])          # the naive path is shifted by one row
    aligned = pipeline.align_like(cond, clean)
    assert np.array_equal(subset(aligned, idx)["ids"], subset(clean, idx)["ids"])
    assert np.array_equal(subset(aligned, idx)["emotion"], subset(clean, idx)["emotion"])
    with pytest.raises(ValueError): pipeline.align_like(subset(cond, np.arange(0, 30)), clean)


def test_corruption_run_works_when_utterances_were_excluded(cfg):
    f = cfg["paths"]["features"]
    for name in ["train", "train__audio_noise", "dev", "dev__audio_noise"]: _mark_failed(f"{f}/{name}.npz", 5)
    conf, cls, meta = pipeline.run_experiment(cfg, 0, 42, "val")
    df = pd.DataFrame(conf)
    assert {"audio_noise|matched", "audio_noise|mismatched"} <= set(df["condition"])
    assert meta["excluded_failed_extraction"]["train"] and meta["excluded_failed_extraction"]["dev"]
