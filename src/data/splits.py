"""Dialogue-grouped splits. Utterances of one dialogue never appear in two partitions."""
import numpy as np


def split_dialogues(dialogue_ids, frac_second, seed):
    """Randomly split unique dialogue ids into (first, second) with `frac_second` in the second part."""
    rng = np.random.default_rng(seed)
    d = np.array(sorted(set(int(x) for x in dialogue_ids)))
    rng.shuffle(d)
    k = max(1, int(round(len(d) * frac_second)))
    return set(d[k:].tolist()), set(d[:k].tolist())


def indices_for(dialogue, dset):
    dialogue = np.asarray(dialogue)
    return np.where(np.isin(dialogue, list(dset)))[0]


def make_partitions(train_dialogue, dev_dialogue, cal_frac, split_seed):
    """Returns index arrays: fit/cal (into train), select/temp (into dev)."""
    fit_d, cal_d = split_dialogues(train_dialogue, cal_frac, split_seed)
    sel_d, temp_d = split_dialogues(dev_dialogue, 0.5, split_seed + 1)
    parts = {"fit": indices_for(train_dialogue, fit_d), "cal": indices_for(train_dialogue, cal_d),
             "select": indices_for(dev_dialogue, sel_d), "temp": indices_for(dev_dialogue, temp_d)}
    assert_disjoint_dialogues(train_dialogue, parts["fit"], parts["cal"])
    assert_disjoint_dialogues(dev_dialogue, parts["select"], parts["temp"])
    return parts


def assert_disjoint_dialogues(dialogue, idx_a, idx_b):
    a, b = set(np.asarray(dialogue)[idx_a].tolist()), set(np.asarray(dialogue)[idx_b].tolist())
    if a & b:
        raise AssertionError(f"Dialogue leakage between partitions: {sorted(a & b)[:5]}")
