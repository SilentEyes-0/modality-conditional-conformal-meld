"""End-to-end experiment on cached features: partitions -> standardise -> train (S1,S2,S3) -> temperature ->
conformal calibration (Designs A and B; global-111, global-pooled, conditional) -> evaluation with cluster bootstrap."""
import os
import numpy as np
import pandas as pd

from .constants import PATTERNS, FULL, MODALITIES, PROFILES, pattern_name, EMOTIONS, SENTIMENTS
from .utils import dump_json, ensure_dir, plan_hash, load_json
from .masking import Standardizer, build_input
from .features.store import load_features, subset
from .data.splits import make_partitions
from .models.train import train_fusion
from .calibration.temperature import fit_temperature
from .models.mlp import softmax
from .conformal import core as cp
from .evaluation import metrics as M

TASKS = {"emotion": (0, "emotion", len(EMOTIONS)), "sentiment": (1, "sentiment", len(SENTIMENTS))}


def drop_failed(d):
    """Exclude utterances where ANY modality failed extraction (the study needs all three modalities per utterance).
    Applied identically to every split; the count is recorded in the run metadata."""
    keep = d["ok"].all(1)
    if keep.all(): return d, []
    return subset(d, np.where(keep)[0]), d["ids"][~keep].tolist()


def align_like(d_c, d_ref):
    """Make a corrupted-feature file row-for-row identical (same utterances, same order) to the already-filtered clean data.
    Without this, partition indices computed on the filtered clean array pick the wrong rows after any excluded utterance."""
    out = subset(d_c, np.where(np.isin(d_c["ids"], d_ref["ids"]))[0])
    if not np.array_equal(out["ids"], d_ref["ids"]):
        raise ValueError("corrupted features are not aligned with the clean features (different utterances or order)")
    return out


def _std_set(raw, std):
    return {"feats": std.transform(raw["feats"]), "emotion": raw["emotion"], "sentiment": raw["sentiment"], "dialogue": raw["dialogue"]}


def _logits_by_mask(model, feats_std):
    n = len(next(iter(feats_std.values())))
    return {m: model.logits(build_input(feats_std, np.full(n, m))) for m in PATTERNS}


def _probs(logits_by_mask, task, T=None):
    k = TASKS[task][0]
    return {m: (softmax(l[k] / T) if T else softmax(l[k])) for m, l in logits_by_mask.items()}


def _load_condition(feats_dir, split, condition):
    path = f"{feats_dir}/{split}__{condition}.npz" if condition else f"{feats_dir}/{split}.npz"
    return load_features(path)


def conformal_rows(ctx, cal_probs, test_probs, y_cal, y_test, dia_test, task, variant, designs, condition, rng):
    """All conformal methods for one (task, variant, condition). Returns list of row dicts."""
    plan = ctx["plan"]; B = plan["bootstrap_B"]; K = TASKS[task][2]
    n_cal = len(y_cal); rows = []
    test_masks_mix = cp.design_B_draw(len(y_test), ctx["rng_test"], PROFILES[plan["test_profile"]])
    sA = cp.design_A_scores(cal_probs, y_cal)

    def evaluate(design, alpha, method, qfun, level_of, ngrp_of):
        for m in PATTERNS:                                   # factorial: every pattern on every test utterance
            sets = cp.prediction_sets(test_probs[m], qfun(m))
            r = M.conformal_summary(sets, y_test, dia_test, B, rng)
            rows.append({**ctx["base"], "condition": condition, "task": task, "variant": variant, "design": design,
                         "alpha": alpha, "method": method, "pattern": pattern_name(m), "level": level_of(m),
                         "n_cal_group": ngrp_of(m), **r})
        P = np.stack([test_probs[m] for m in PATTERNS])        # mixture: one pattern per utterance
        pidx = np.array([PATTERNS.index(m) for m in test_masks_mix])
        probs_mix = P[pidx, np.arange(len(y_test))]
        q = np.array([qfun(m) for m in test_masks_mix])
        sets = cp.prediction_sets(probs_mix, q)
        r = M.conformal_summary(sets, y_test, dia_test, B, rng)
        rows.append({**ctx["base"], "condition": condition, "task": task, "variant": variant, "design": design,
                     "alpha": alpha, "method": method, "pattern": "mixture", "level": "-", "n_cal_group": n_cal, **r})
        for c, (n_c, cov_c) in M.class_conditional(sets, y_test, K).items():
            rows.append({**ctx["base"], "condition": condition, "task": task, "variant": variant, "design": design,
                         "alpha": alpha, "method": method, "pattern": f"class{c}", "n": n_c, "coverage": cov_c})

    for alpha in plan["alphas"]:
        if "A" in designs:
            pool_masks = cp.design_B_draw(n_cal, ctx["rng_cal"], PROFILES["uniform"])
            pooled_scores = np.array([sA[m][i] for i, m in enumerate(pool_masks)])
            q111 = cp.global_111(sA, alpha); qpool = cp.global_pooled(pooled_scores, alpha)
            qm = {m: cp.conformal_quantile(sA[m], alpha) for m in PATTERNS}
            evaluate("A", alpha, "global_111", lambda m: q111, lambda m: "g", lambda m: int(n_cal))
            evaluate("A", alpha, "global_pooled", lambda m: qpool, lambda m: "g", lambda m: int(n_cal))
            evaluate("A", alpha, "conditional", lambda m: qm[m], lambda m: 0, lambda m: int(n_cal))
        if "B" in designs:
            for prof in plan["cal_profiles"]:
                masks_b = cp.design_B_draw(n_cal, ctx["rng_cal"], PROFILES[prof])
                sB = np.array([sA[m][i] for i, m in enumerate(masks_b)])
                q111 = cp.conformal_quantile(sB[masks_b == FULL], alpha); qpool = cp.global_pooled(sB, alpha)
                for n_min in plan["n_min_list"]:
                    tab = cp.mondrian(sB, masks_b, alpha, n_min)
                    d = f"B-{prof}-nmin{n_min}"
                    evaluate(d, alpha, "global_111", lambda m: q111, lambda m: "g", lambda m: int((masks_b == FULL).sum()))
                    evaluate(d, alpha, "global_pooled", lambda m: qpool, lambda m: "g", lambda m: int(n_cal))
                    evaluate(d, alpha, "conditional", lambda m: tab[m][0], lambda m: tab[m][1], lambda m: tab[m][2])
                    for tax in plan.get("taxonomies", []):          # ablation (c): coarser taxonomies, Design B only
                        tt = cp.mondrian_taxonomy(sB, masks_b, alpha, n_min, tax)
                        evaluate(d, alpha, f"conditional_{tax}", lambda m, tt=tt: tt[m][0], lambda m, tt=tt: tt[m][1], lambda m, tt=tt: tt[m][2])
    return rows


def run_experiment(cfg, seed, split_seed, eval_split="val", out_dir=None):
    """eval_split: 'val' (development only: dev select subset) or 'test' (final evaluation)."""
    plan = cfg["analysis_plan"]; tcfg = cfg["train"]; feats_dir = cfg["paths"]["features"]
    out_dir = ensure_dir(out_dir or cfg["paths"]["outputs"])
    loaded = [drop_failed(load_features(f"{feats_dir}/{s}.npz")) for s in ["train", "dev", "test"]]
    (tr, dv, te) = (x[0] for x in loaded); excluded = {s: x[1] for s, x in zip(["train", "dev", "test"], loaded)}
    parts = make_partitions(tr["dialogue"], dv["dialogue"], cfg["split"]["cal_frac"], split_seed)
    keep = plan.get("cal_keep_frac", 1.0)                      # ablation (d): subsample the calibration set by dialogue
    if keep < 1.0:
        dl = np.unique(tr["dialogue"][parts["cal"]]); r2 = np.random.default_rng(split_seed + 5)
        chosen = r2.choice(dl, size=max(1, int(round(len(dl) * keep))), replace=False)
        parts["cal"] = parts["cal"][np.isin(tr["dialogue"][parts["cal"]], chosen)]
    raw = {"fit": subset(tr, parts["fit"]), "cal": subset(tr, parts["cal"]), "select": subset(dv, parts["select"]),
           "temp": subset(dv, parts["temp"]), "test": te}
    std = Standardizer().fit(raw["fit"]["feats"])
    S = {k: _std_set(v, std) for k, v in raw.items()}
    ev_key = "test" if eval_split == "test" else "select"
    base = {"seed": seed, "split_seed": split_seed, "eval_split": eval_split}
    rng = np.random.default_rng(seed + 1000)
    ctx = {"plan": plan, "base": base, "rng_cal": np.random.default_rng(seed + 11), "rng_test": np.random.default_rng(seed + 22)}

    # ---- train
    m_drop, h3 = train_fusion(S["fit"], S["select"], "dropout", tcfg, seed)
    m_full, h2 = train_fusion(S["fit"], S["select"], "full", tcfg, seed)
    uni = {}
    for name, mask in zip(MODALITIES, [4, 2, 1]):
        uni[name], _ = train_fusion(S["fit"], S["select"], ("single", mask), tcfg, seed)

    # ---- temperature (S4): one per task, temperature-fit subset under the uniform mask mixture
    tl = _logits_by_mask(m_drop, S["temp"]["feats"]); n_t = len(S["temp"]["emotion"])
    mix = cp.design_B_draw(n_t, np.random.default_rng(seed + 33), PROFILES["uniform"])
    Ts = {}
    for task, (k, key, _) in TASKS.items():
        L = np.stack([tl[m][k] for m in PATTERNS])
        L = L[[PATTERNS.index(m) for m in mix], np.arange(n_t)]
        Ts[task] = fit_temperature(L, S["temp"][key])

    # ---- classification metrics (S1, S2, S3 raw/scaled)
    ev = S[ev_key]; n_ev = len(ev["emotion"]); cls_rows = []
    lg = {"S3": _logits_by_mask(m_drop, ev["feats"]), "S2": _logits_by_mask(m_full, ev["feats"])}
    for task, (k, key, _) in TASKS.items():
        y = ev[key]
        for sysname, L in lg.items():
            for m in PATTERNS:
                pr = softmax(L[m][k]); r = M.classification_metrics(pr, y, plan["ece_bins"])
                row = {**base, "system": sysname, "task": task, "pattern": pattern_name(m), **{f"{a}_R": b for a, b in r.items()}}
                if sysname == "S3":
                    r2 = M.classification_metrics(softmax(L[m][k] / Ts[task]), y, plan["ece_bins"])
                    row.update({f"{a}_T": b for a, b in r2.items()})
                cls_rows.append(row)
        for name, mask in zip(MODALITIES, [4, 2, 1]):
            L = uni[name].logits(build_input(ev["feats"], np.full(n_ev, mask)))
            r = M.classification_metrics(softmax(L[k]), y, plan["ece_bins"])
            cls_rows.append({**base, "system": f"S1-{name}", "task": task, "pattern": pattern_name(mask), **{f"{a}_R": b for a, b in r.items()}})

    # ---- conformal (S5-S7), clean condition
    cal_lg = _logits_by_mask(m_drop, S["cal"]["feats"]); test_lg = lg["S3"]
    conf_rows = []
    for task, (k, key, _) in TASKS.items():
        for variant, T in [("R", None), ("T", Ts[task])]:
            conf_rows += conformal_rows(ctx, _probs(cal_lg, task, T), _probs(test_lg, task, T), S["cal"][key], ev[key],
                                        ev["dialogue"], task, variant, plan["designs"], "clean", rng)

    # ---- corruption (matched / mismatched calibration), design A only
    for c in plan.get("corruption", []):
        name, mod, calib = c["name"], c["modality"], c.get("calibration", "mismatched")
        te_c = align_like(_load_condition(feats_dir, "test", name), te) if ev_key == "test" else None
        if te_c is None:
            dv_c = align_like(_load_condition(feats_dir, "dev", name), dv); ev_raw_c = subset(dv_c, parts["select"])
        else:
            ev_raw_c = te_c
        ev_c = _std_set(ev_raw_c, std)
        cal_src = S["cal"]
        if calib == "matched":
            tr_c = align_like(_load_condition(feats_dir, "train", name), tr); cal_src = _std_set(subset(tr_c, parts["cal"]), std)
        cal_l = _logits_by_mask(m_drop, cal_src["feats"]); te_l = _logits_by_mask(m_drop, ev_c["feats"])
        for task, (k, key, _) in TASKS.items():
            for variant, T in [("R", None), ("T", Ts[task])]:
                conf_rows += conformal_rows(ctx, _probs(cal_l, task, T), _probs(te_l, task, T), cal_src[key], ev_c[key],
                                            ev_c["dialogue"], task, variant, ["A"], f"{name}|{calib}", rng)

    meta = {**base, "temperatures": Ts, "n": {k: int(len(v["emotion"])) for k, v in S.items()},
            "n_params": m_drop.n_params(), "train_hist_S3": h3, "train_hist_S2": h2,
            "cal_class_counts_emotion": np.bincount(S["cal"]["emotion"], minlength=7).tolist(),
            "excluded_failed_extraction": excluded, "plan_hash": plan_hash(plan)}
    tag = f"seed{seed}_split{split_seed}_{eval_split}"
    pd.DataFrame(conf_rows).to_csv(f"{out_dir}/conformal_{tag}.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(f"{out_dir}/classification_{tag}.csv", index=False)
    dump_json(meta, f"{out_dir}/meta_{tag}.json")
    return conf_rows, cls_rows, meta


# ------------------------------------------------------------------ plan lock
def freeze_plan(cfg, path=None):
    path = path or os.path.join(cfg["paths"]["outputs"], "plan_lock.json")
    dump_json({"plan_hash": plan_hash(cfg["analysis_plan"]), "plan": cfg["analysis_plan"], "test_evaluated": False}, path)
    return path


def check_lock_for_test(cfg, force=False):
    path = os.path.join(cfg["paths"]["outputs"], "plan_lock.json")
    if not os.path.exists(path):
        raise RuntimeError("Final test evaluation requires a frozen plan: run `python -m src.run freeze` first.")
    lock = load_json(path)
    if lock["plan_hash"] != plan_hash(cfg["analysis_plan"]):
        raise RuntimeError("analysis_plan changed after freezing; restore it or document the deviation and re-freeze.")
    if lock.get("test_evaluated") and not force:
        raise RuntimeError("Test set already evaluated for this plan. Use --force only to document a deviation.")
    return path, lock


def mark_test_evaluated(path, lock):
    lock["test_evaluated"] = True
    dump_json(lock, path)
