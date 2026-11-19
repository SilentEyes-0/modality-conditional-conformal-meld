"""Estimate extraction time BEFORE committing to a long run: time n and 2n clips and extrapolate linearly.
Using the difference removes one-off model-loading time. This is an ESTIMATE; real time can vary with disk and CPU load.
Usage:  python -m src.features.timing --config configs/experiment.yaml --split train --n 50
"""
import argparse, os, time


def extrapolate(t_n, t_2n, n, total):
    """Seconds for `total` items given the time for n items and for 2n items (per-item cost = (t_2n - t_n) / n)."""
    if n <= 0 or total < 0: raise ValueError("n must be positive and total non-negative")
    per_item = max(t_2n - t_n, 0.0) / n
    return per_item * total, per_item


def fmt(seconds):
    h = seconds / 3600
    return f"{seconds:.0f} s (~{h:.1f} h)"


def main():
    from ..utils import load_config
    from ..data.meld import load_split_csv, media_path
    from . import extract as E
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--split", default="train"); ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args(); cfg = load_config(a.config)
    total = sum(len(load_split_csv(os.path.join(cfg["data"]["root"], cfg["data"]["csv"][s]))) for s in ["train", "dev", "test"])
    df = load_split_csv(os.path.join(cfg["data"]["root"], cfg["data"]["csv"][a.split]))
    paths = [media_path(cfg, a.split, u, "video") for u in df["id"]]
    fns = {"text": lambda k: E.encode_text(df["text"].tolist()[:k], cfg),
           "audio": lambda k: E.encode_audio(paths[:k], cfg), "video": lambda k: E.encode_video(paths[:k], cfg)}
    grand = 0.0
    for name, fn in fns.items():
        t0 = time.time(); fn(a.n); t1 = time.time() - t0
        t0 = time.time(); fn(2 * a.n); t2 = time.time() - t0
        est, per = extrapolate(t1, t2, a.n, total); grand += est
        print(f"{name}: {per:.3f} s per item -> about {fmt(est)} for all {total} utterances (estimate)")
    print(f"ALL MODALITIES: about {fmt(grand)} (estimate; ignores failures and disk effects)")


if __name__ == "__main__":
    main()
