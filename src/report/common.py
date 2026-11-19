"""Shared helpers for the P8 report scripts. Reads only the *_test.csv summary files."""
import os
import numpy as np
import pandas as pd

PATTERN_ORDER = ["111", "110", "101", "011", "100", "010", "001"]
MAIN_METHODS = ["conditional", "global_111", "global_pooled"]
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENTS = ["negative", "neutral", "positive"]
ROW_KEEP = PATTERN_ORDER + ["mixture"]  # real patterns plus the overall test-mixture row


def keep_pattern_rows(df, col="pattern"):
    """Drop the per-class rows (class0, class1, ...); keep the seven patterns and the mixture row."""
    return df[df[col].isin(ROW_KEEP)]


def class_name(task, label):
    names = EMOTIONS if task == "emotion" else SENTIMENTS
    i = int(str(label).replace("class", ""))
    return names[i] if i < len(names) else str(label)


def load(d):
    s = pd.read_csv(os.path.join(d, "summary_conformal_test.csv"), dtype={"pattern": str})
    c = pd.read_csv(os.path.join(d, "summary_classification_test.csv"), dtype={"pattern": str})
    w = pd.read_csv(os.path.join(d, "worst_pattern_test.csv"))
    return s, c, w


def sort_patterns(df, col="pattern"):
    """Real patterns first, in a fixed order; any other rows (e.g. an overall row) afterwards."""
    key = df[col].map(lambda p: PATTERN_ORDER.index(p) if p in PATTERN_ORDER else 100)
    return df.assign(_k=key, _s=df[col]).sort_values(["_k", "_s"]).drop(columns=["_k", "_s"])


def pm(mean, std, nd=3):
    if pd.isna(mean):
        return ""
    return f"{mean:.{nd}f}" if pd.isna(std) else f"{mean:.{nd}f} ± {std:.{nd}f}"


def fmt_cell(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def to_markdown(df, nd=3):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt_cell(v, nd) for v in r.tolist()) + " |")
    return "\n".join(lines) + "\n"


def save(df, out_dir, name, nd=3):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(df, nd))
    print("wrote", name, df.shape)


def split_condition(cond):
    """'audio_snr10|matched' -> ('audio_snr10', 'matched'); 'clean' -> ('clean', '')."""
    if "|" in cond:
        a, b = cond.split("|", 1)
        return a, b
    return cond, ""


def split_design_b(design):
    """'B-skewed-nmin50' -> ('skewed', 50)."""
    parts = design.split("-")
    return parts[1], int(parts[2].replace("nmin", ""))


def wilson(p, n, z=1.96):
    """Approximate 95% binomial interval; ignores that utterances within a dialogue are dependent."""
    if n is None or n <= 0 or pd.isna(p):
        return (np.nan, np.nan)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
