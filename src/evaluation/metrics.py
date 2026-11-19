import numpy as np
from sklearn.metrics import f1_score


def ece(probs, y, bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        i = (conf > lo) & (conf <= hi)
        if i.any(): e += i.mean() * abs(acc[i].mean() - conf[i].mean())
    return float(e)


def brier(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def nll(probs, y):
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def classification_metrics(probs, y, bins=15):
    pred = probs.argmax(1); K = probs.shape[1]; labels = list(range(K))
    return {"acc": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
            "ece": ece(probs, y, bins), "brier": brier(probs, y), "nll": nll(probs, y)}


def set_stats(sets, y):
    sz = sets.sum(1); cov = sets[np.arange(len(y)), y].astype(float)
    return cov, sz


def cluster_bootstrap(per_utt_values, dialogue, B, rng, alpha=0.05):
    """Dialogue-level cluster bootstrap CI for the mean of each column of per_utt_values (N, c)."""
    vals = np.asarray(per_utt_values, float)
    if vals.ndim == 1: vals = vals[:, None]
    d_ids, inv = np.unique(dialogue, return_inverse=True)
    D = len(d_ids)
    sums = np.zeros((D, vals.shape[1])); np.add.at(sums, inv, vals)
    cnt = np.bincount(inv, minlength=D).astype(float)
    idx = rng.integers(0, D, size=(B, D))
    est = sums[idx].sum(1) / cnt[idx].sum(1)[:, None]
    lo, hi = np.quantile(est, [alpha / 2, 1 - alpha / 2], axis=0)
    return lo, hi


def conformal_summary(sets, y, dialogue, B, rng):
    cov, sz = set_stats(sets, y)
    vals = np.stack([cov, sz], 1)
    lo, hi = cluster_bootstrap(vals, dialogue, B, rng)
    return {"n": int(len(y)), "coverage": float(cov.mean()), "cov_lo": float(lo[0]), "cov_hi": float(hi[0]),
            "size": float(sz.mean()), "size_lo": float(lo[1]), "size_hi": float(hi[1]),
            "empty": float((sz == 0).mean()), "singleton": float((sz == 1).mean()), "full": float((sz == sets.shape[1]).mean())}


def class_conditional(sets, y, K):
    cov = sets[np.arange(len(y)), y].astype(float)
    return {c: (int((y == c).sum()), float(cov[y == c].mean()) if (y == c).any() else float("nan")) for c in range(K)}
