"""Synthetic MELD-shaped features for smoke tests ONLY.
It is NOT a model of MELD; any number computed from it says nothing about MELD or about the method's real behaviour."""
import numpy as np
from ..constants import DIMS, MODALITIES
from ..features.store import save_features

_PRIOR = np.array([0.11, 0.027, 0.027, 0.175, 0.47, 0.068, 0.12])
PRIOR = _PRIOR / _PRIOR.sum()                 # MELD-like imbalance (illustrative)
SENT_OF = np.array([0, 0, 0, 2, 1, 0, 2])     # emotion -> sentiment (illustrative mapping)
SPLIT_OFFSET = {"train": 0, "dev": 10000, "test": 20000}
SPLIT_SEED = {"train": 1, "dev": 2, "test": 3}


def make_split(split, n_dialogues, seed=0, strength=(1.6, 0.9, 0.6), noise=1.0):
    rng = np.random.default_rng(seed * 100 + SPLIT_SEED[split])
    off = SPLIT_OFFSET[split]
    ids, dia, em = [], [], []
    for d in range(n_dialogues):
        L = int(rng.integers(3, 12))
        mood = rng.dirichlet(np.ones(7) * 2 + PRIOR * 10)       # dialogue-level dependence
        p = 0.5 * mood + 0.5 * PRIOR
        p = p / p.sum()
        for u in range(L):
            ids.append(f"dia{off + d}_utt{u}"); dia.append(off + d); em.append(rng.choice(7, p=p))
    em = np.array(em); sent = SENT_OF[em]
    feats = {}
    for k, (m, s) in enumerate(zip(MODALITIES, strength)):
        proto = np.random.default_rng(seed * 100 + 50 + k).normal(size=(7, DIMS[m]))   # shared across splits
        feats[m] = (s * proto[em] + noise * rng.normal(size=(len(em), DIMS[m]))).astype(np.float32)
    return ids, np.array(dia), feats, em, sent


def write_synthetic_dataset(out_dir, seed=0, sizes=(160, 24, 48)):
    counts = {}
    for split, nd in zip(["train", "dev", "test"], sizes):
        ids, dia, feats, em, sent = make_split(split, nd, seed)
        save_features(f"{out_dir}/{split}.npz", ids, dia, feats, em, sent)
        counts[split] = len(ids)
    return counts


def corrupt_modality(out_dir, split, modality, name, sigma, seed=0):
    """Synthetic 'corruption': add feature noise. Real corruption is done on raw signals (see features/corruption.py)."""
    from ..features.store import load_features
    d = load_features(f"{out_dir}/{split}.npz")
    rng = np.random.default_rng(seed + 777)
    feats = {m: d["feats"][m].copy() for m in MODALITIES}
    feats[modality] = feats[modality] + sigma * rng.normal(size=feats[modality].shape).astype(np.float32)
    save_features(f"{out_dir}/{split}__{name}.npz", d["ids"], d["dialogue"], feats, d["emotion"], d["sentiment"])
