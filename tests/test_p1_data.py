import numpy as np, pandas as pd, pytest
from src.data.splits import make_partitions, assert_disjoint_dialogues
from src.data.meld import load_split_csv
from src.data.synthetic import write_synthetic_dataset, make_split
from src.features.store import load_features, subset
from src.constants import EMOTIONS, D_IN, DIMS

def test_partitions_are_dialogue_disjoint():
    rng = np.random.default_rng(0)
    tr = np.repeat(np.arange(100), rng.integers(3, 10, size=100)); dv = 1000 + np.repeat(np.arange(20), 5)
    parts = make_partitions(tr, dv, 0.2, 42)
    assert_disjoint_dialogues(tr, parts["fit"], parts["cal"]); assert_disjoint_dialogues(dv, parts["select"], parts["temp"])
    assert len(parts["fit"]) + len(parts["cal"]) == len(tr)
    assert abs(len(set(tr[parts["cal"]].tolist())) - 20) <= 1
    with pytest.raises(AssertionError):
        assert_disjoint_dialogues(tr, np.array([0, 1, 2]), np.array([2, 3]))


def test_meld_csv_label_mapping(tmp_path):
    df = pd.DataFrame({"Utterance": ["hi", "no"], "Emotion": ["joy", "Anger"], "Sentiment": ["positive", "negative"],
                       "Dialogue_ID": [0, 0], "Utterance_ID": [0, 1]})
    p = tmp_path / "x.csv"; df.to_csv(p, index=False)
    out = load_split_csv(str(p)); assert out["emotion"].tolist() == [EMOTIONS.index("joy"), EMOTIONS.index("anger")]
    assert out["id"].tolist() == ["dia0_utt0", "dia0_utt1"]


def test_synthetic_dataset_shapes_and_determinism(tmp_path):
    a = write_synthetic_dataset(str(tmp_path / "a"), seed=0, sizes=(20, 5, 8)); b = write_synthetic_dataset(str(tmp_path / "b"), seed=0, sizes=(20, 5, 8))
    assert a == b
    da, db = load_features(str(tmp_path / "a" / "train.npz")), load_features(str(tmp_path / "b" / "train.npz"))
    assert da["text"].shape[1] == DIMS["text"] and da["audio"].shape[1] == 88 and da["video"].shape[1] == 512
    assert np.array_equal(da["text"], db["text"]) and np.array_equal(da["emotion"], db["emotion"])
    assert not set(da["dialogue"].tolist()) & set(load_features(str(tmp_path / "a" / "test.npz"))["dialogue"].tolist())
    s = subset(da, np.arange(5)); assert len(s["ids"]) == 5 and s["feats"]["video"].shape == (5, 512)
