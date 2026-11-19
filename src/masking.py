"""Availability masks, standardisation and the 1,371-dimensional fused input."""
import numpy as np
from .constants import MODALITIES, PATTERNS


def mask_bits(mask):
    mask = np.asarray(mask)
    return np.stack([(mask >> 2) & 1, (mask >> 1) & 1, mask & 1], axis=1).astype(np.float32)


class Standardizer:
    """Per-modality mean/std fitted on the model-fit subset ONLY, then applied unchanged."""

    def fit(self, feats):
        self.mu = {m: feats[m].mean(0) for m in MODALITIES}
        self.sd = {m: np.maximum(feats[m].std(0), 1e-6) for m in MODALITIES}
        return self

    def transform(self, feats):
        return {m: ((feats[m] - self.mu[m]) / self.sd[m]).astype(np.float32) for m in MODALITIES}


def build_input(feats_std, mask):
    """Zero absent branches AFTER standardisation, then append the three mask bits."""
    bits = mask_bits(mask)
    parts = [feats_std[m] * bits[:, i:i + 1] for i, m in enumerate(MODALITIES)]
    return np.concatenate(parts + [bits], axis=1).astype(np.float32)


def sample_masks(n, rng, probs=None):
    """Sample masks from a distribution over the seven non-empty patterns (never 000)."""
    probs = probs or {p: 1 / 7 for p in PATTERNS}
    pats = np.array(list(probs.keys()))
    p = np.array(list(probs.values()), dtype=float)
    return rng.choice(pats, size=n, p=p / p.sum())
