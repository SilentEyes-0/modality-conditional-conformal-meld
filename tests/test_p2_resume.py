import numpy as np, pandas as pd, pytest
from src.features import extract as E
from src.features.check import check_features
from src.features.store import load_features
from src.data.synthetic import make_split
from src.constants import DIMS

N = 23


def _make(tmp_path, monkeypatch, crash_state):
    df = pd.DataFrame({"Utterance": [f"t{i}" for i in range(N)], "Emotion": ["joy"] * N, "Sentiment": ["positive"] * N,
                       "Dialogue_ID": [i // 5 for i in range(N)], "Utterance_ID": [i % 5 for i in range(N)]})
    (tmp_path / "x.csv").write_text(df.to_csv(index=False))
    cfg = {"data": {"root": str(tmp_path), "csv": {"train": "x.csv"}, "video_dir": {"train": "v"}},
           "paths": {"features": str(tmp_path / "feats")}, "features": {"chunk_size": 5}}
    calls = []

    def fake(mod):
        def f(items, cfg_, corrupt=None, rng=None):
            calls.append((mod, len(items)))
            if mod == "video" and crash_state["on"] and sum(1 for c in calls if c[0] == "video") == 3: raise RuntimeError("simulated crash")
            return np.full((len(items), DIMS[mod]), 1.0, np.float32), np.ones(len(items), bool)
        return f
    for mod in ["text", "audio", "video"]: monkeypatch.setattr(E, f"encode_{mod}", fake(mod))
    return cfg, calls


def test_interrupted_run_resumes_and_equals_uninterrupted(tmp_path, monkeypatch):
    state = {"on": True}; (tmp_path / "a").mkdir()
    cfg, calls = _make(tmp_path / "a", monkeypatch, state)
    with pytest.raises(RuntimeError): E.run(cfg, "train", "all")
    first_video_chunks = sum(1 for c in calls if c[0] == "video")
    state["on"] = False; calls.clear()
    E.run(cfg, "train", "all")                                                      # rerun the same command
    assert [c for c in calls if c[0] in ("text", "audio")] == []                    # finished modalities are skipped entirely
    assert sum(1 for c in calls if c[0] == "video") == 5 - (first_video_chunks - 1)  # only unfinished video chunks are redone (5 chunks of 5, 5, 5, 5, 3)
    E.assemble(cfg, "train")
    d = load_features(cfg["paths"]["features"] + "/train.npz")
    assert d["text"].shape == (N, 768) and d["audio"].shape == (N, 88) and d["video"].shape == (N, 512) and d["ok"].all()
    rep = check_features(d); assert rep["problems"] == [] and rep["n"] == N


def test_failed_items_are_flagged_and_zero_filled():
    ids, dia, feats, em, sent = make_split("train", 5, 0)
    ok = np.ones((len(ids), 3), bool); ok[2, 1] = False; feats["audio"][2] = 0
    rep = check_features({"ids": np.array(ids), "feats": feats, "ok": ok})
    assert rep["failed_ids"] == [ids[2]] and rep["problems"] == []
    feats["audio"][2] = 1.0
    rep = check_features({"ids": np.array(ids), "feats": feats, "ok": ok})
    assert any("not zero-filled" in p for p in rep["problems"])
