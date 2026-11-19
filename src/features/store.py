import os
import numpy as np
from ..constants import MODALITIES


def save_features(path, ids, dialogue, feats, emotion, sentiment, ok=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(ids)
    ok = np.ones((n, 3), bool) if ok is None else ok
    np.savez_compressed(path, ids=np.asarray(ids, dtype="<U32"), dialogue=np.asarray(dialogue, dtype=np.int64),
                        text=feats["text"].astype(np.float32), audio=feats["audio"].astype(np.float32),
                        video=feats["video"].astype(np.float32), emotion=np.asarray(emotion, dtype=np.int64),
                        sentiment=np.asarray(sentiment, dtype=np.int64), ok=ok)


def load_features(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["feats"] = {m: d[m] for m in MODALITIES}
    return d


def subset(d, idx):
    out = {k: d[k][idx] for k in ["ids", "dialogue", "emotion", "sentiment", "ok"]}
    out["feats"] = {m: d["feats"][m][idx] for m in MODALITIES}
    return out
