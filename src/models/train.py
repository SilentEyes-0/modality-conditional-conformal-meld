"""Training of unimodal baselines (S1), full-input fusion (S2) and modality-dropout fusion (S3).
Model selection: validation negative log-likelihood under a FIXED validation mask draw."""
import numpy as np
from ..constants import PATTERNS, FULL, D_IN
from ..masking import build_input, sample_masks
from .mlp import FusionMLP, Adam, loss_and_grads, softmax


def rho_probs(rho_full):
    others = [p for p in PATTERNS if p != FULL]
    d = {FULL: rho_full}; d.update({p: (1 - rho_full) / len(others) for p in others})
    return d


def _masks(mode, n, rng, rho_full):
    if mode == "dropout": return sample_masks(n, rng, rho_probs(rho_full))
    if mode == "full": return np.full(n, FULL)
    if isinstance(mode, tuple) and mode[0] == "single": return np.full(n, mode[1])
    raise ValueError(mode)


def train_fusion(fit, sel, mode, tcfg, seed):
    """fit/sel: dicts with 'feats' (standardised), 'emotion', 'sentiment'. Returns (model, history)."""
    rng = np.random.default_rng(seed)
    model = FusionMLP(D_IN, tcfg["h1"], tcfg["h2"], dropout=tcfg["dropout"], seed=seed)
    opt = Adam(model.p, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    n = len(fit["emotion"]); lam = tcfg["lambda_sentiment"]
    w_e = None
    if tcfg.get("class_weighted"):
        cnt = np.bincount(fit["emotion"], minlength=7).astype(float) + 1
        w_e = (cnt.sum() / (7 * cnt)); w_e = w_e / w_e.mean()
    val_rng = np.random.default_rng(10_000 + seed)
    Xval = build_input(sel["feats"], _masks(mode if mode != "dropout" else "dropout", len(sel["emotion"]), val_rng, 1 / 7 if mode == "dropout" else 0))
    best, best_params, bad, hist = np.inf, model.get_params(), 0, []
    for ep in range(tcfg["epochs"]):
        perm = rng.permutation(n)
        for i in range(0, n, tcfg["batch_size"]):
            b = perm[i:i + tcfg["batch_size"]]
            masks = _masks(mode, len(b), rng, tcfg["rho_full"])
            X = build_input({k: v[b] for k, v in fit["feats"].items()}, masks)
            _, g = loss_and_grads(model, X, fit["emotion"][b], fit["sentiment"][b], lam, w_e, True, rng)
            opt.step(model.p, g)
        vl, _ = loss_and_grads(model, Xval, sel["emotion"], sel["sentiment"], lam, None, False)
        hist.append(float(vl))
        if vl < best - 1e-5: best, best_params, bad = vl, model.get_params(), 0
        else:
            bad += 1
            if bad >= tcfg["patience"]: break
    model.set_params(best_params)
    return model, {"val_nll": hist, "best_val_nll": float(best), "epochs_run": len(hist)}


def logits_under_masks(model, feats_std, masks):
    return model.logits(build_input(feats_std, masks))
