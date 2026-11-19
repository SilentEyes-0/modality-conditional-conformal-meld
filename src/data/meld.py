"""MELD CSV loading, validation and inspection. Paths come from the config (verify them against your download)."""
import os
import numpy as np
import pandas as pd
from ..constants import EMOTIONS, SENTIMENTS, EXPECTED_COUNTS


def load_split_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Dialogue_ID": "dialogue", "Utterance_ID": "utt", "Emotion": "emotion_name",
                            "Sentiment": "sentiment_name", "Utterance": "text"})
    df["id"] = "dia" + df["dialogue"].astype(str) + "_utt" + df["utt"].astype(str)
    df["emotion"] = df["emotion_name"].str.lower().map({e: i for i, e in enumerate(EMOTIONS)})
    df["sentiment"] = df["sentiment_name"].str.lower().map({s: i for i, s in enumerate(SENTIMENTS)})
    if df["emotion"].isna().any() or df["sentiment"].isna().any():
        bad = df[df["emotion"].isna() | df["sentiment"].isna()]["id"].tolist()[:5]
        raise ValueError(f"Unknown label names in {csv_path}; first offending ids: {bad}")
    if df["id"].duplicated().any():
        raise ValueError(f"Duplicate utterance ids in {csv_path}")
    return df[["id", "dialogue", "utt", "text", "emotion", "sentiment"]].reset_index(drop=True)


def media_path(cfg, split, uid, kind):
    root = cfg["data"]["root"]
    sub = cfg["data"]["video_dir"][split]
    return os.path.join(root, sub, f"{uid}.mp4")


def inspect_split(cfg, split):
    df = load_split_csv(os.path.join(cfg["data"]["root"], cfg["data"]["csv"][split]))
    missing = [u for u in df["id"] if not os.path.exists(media_path(cfg, split, u, "video"))]
    exp_d, exp_u = EXPECTED_COUNTS[split]
    return {
        "split": split, "dialogues": int(df["dialogue"].nunique()), "utterances": int(len(df)),
        "expected_dialogues": exp_d, "expected_utterances": exp_u,
        "counts_match_official": bool(df["dialogue"].nunique() == exp_d and len(df) == exp_u),
        "emotion_counts": {EMOTIONS[i]: int((df["emotion"] == i).sum()) for i in range(len(EMOTIONS))},
        "sentiment_counts": {SENTIMENTS[i]: int((df["sentiment"] == i).sum()) for i in range(len(SENTIMENTS))},
        "missing_media": missing[:50], "n_missing_media": len(missing),
    }
