"""Split conformal prediction with score s = 1 - p(y|x): global-111, global-pooled and Mondrian-by-mask with fallback."""
import numpy as np
from ..constants import FULL, PATTERNS, cardinality


def conformal_quantile(scores, alpha):
    """k = ceil((n+1)(1-alpha)); the k-th smallest score, or +inf if k > n (group too small -> full label set)."""
    s = np.sort(np.asarray(scores, float)); n = s.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if (n == 0 or k > n) else float(s[k - 1])


def scores_from_probs(probs, y):
    return 1.0 - probs[np.arange(len(y)), y]


def global_111(scores_by_mask, alpha):
    """Threshold from complete-modality calibration records only (stress baseline)."""
    return conformal_quantile(scores_by_mask[FULL], alpha)


def global_pooled(scores, alpha):
    """Threshold from scores pooled over the calibration mask mixture (fair baseline)."""
    return conformal_quantile(scores, alpha)


def mondrian(scores, masks, alpha, n_min):
    """Per-pattern thresholds with pre-specified fallback: level 0 exact pattern (n_m >= n_min);
    level 1 same-cardinality group (n >= n_min); level 2 global pooled. Decisions use COUNTS only.
    Returns {pattern: (threshold, level, n_used)}."""
    scores, masks = np.asarray(scores), np.asarray(masks)
    card = np.array([cardinality(m) for m in masks])
    glob = conformal_quantile(scores, alpha); out = {}
    for m in PATTERNS:
        i = masks == m
        if i.sum() >= n_min: out[m] = (conformal_quantile(scores[i], alpha), 0, int(i.sum())); continue
        g = card == cardinality(m)
        if g.sum() >= n_min: out[m] = (conformal_quantile(scores[g], alpha), 1, int(g.sum()))
        else: out[m] = (glob, 2, int(len(scores)))
    return out


TAXONOMIES = {"exact": lambda m: int(m), "cardinality": cardinality, "text": lambda m: (int(m) >> 2) & 1}


def mondrian_taxonomy(scores, masks, alpha, n_min, taxonomy):
    """Coarser Mondrian taxonomies for the ablation: group by number of modalities or by text present/absent.
    A group with fewer than n_min examples falls back to the global pooled threshold (level 2)."""
    key = TAXONOMIES[taxonomy]; scores, masks = np.asarray(scores), np.asarray(masks)
    keys = np.array([key(m) for m in masks]); glob = conformal_quantile(scores, alpha); out = {}
    for m in PATTERNS:
        g = keys == key(m)
        out[m] = (conformal_quantile(scores[g], alpha), 0, int(g.sum())) if g.sum() >= n_min else (glob, 2, int(len(scores)))
    return out


def prediction_sets(probs, q):
    """Boolean (N, K) membership: y in C iff 1 - p(y|x) <= q (q may be a scalar or per-example array)."""
    q = np.asarray(q, float)
    if q.ndim == 0: q = np.full(len(probs), float(q))
    return (1.0 - probs) <= q[:, None]


def design_A_scores(probs_by_mask, y):
    """Design A: every calibration utterance under every pattern -> {pattern: scores (N,)}."""
    return {m: scores_from_probs(probs_by_mask[m], y) for m in PATTERNS}


def design_B_draw(n, rng, profile):
    """Design B: one pattern per calibration utterance drawn from the frequency profile pi."""
    pats = np.array(list(profile.keys())); p = np.array(list(profile.values()), float)
    return rng.choice(pats, size=n, p=p / p.sum())
