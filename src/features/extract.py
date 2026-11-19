"""Frozen feature extraction (text: RoBERTa-base; audio: openSMILE eGeMAPSv02; video: ResNet-18 on 8 frames).
Heavy libraries are imported lazily. NOT executed in the build environment (no MELD data or model weights there):
run it on your machine and check the log for failures. Failed items are zero-filled and flagged ok=False, never fabricated.

  python -m src.features.extract --config configs/experiment.yaml --split train --modality all
  python -m src.features.extract --config ... --split test --modality audio --condition audio_snr10 --corrupt audio:snr=10
  python -m src.features.extract assemble --config ... --split train [--condition NAME]
"""
import argparse, os, subprocess, tempfile
import numpy as np
from ..constants import DIMS, MODALITIES
from ..utils import load_config, ensure_dir
from ..data.meld import load_split_csv, media_path
from .store import save_features, load_features
from . import corruption as C


def parts_dir(cfg, condition):
    return ensure_dir(os.path.join(cfg["paths"]["features"], "parts", condition or "clean"))


def encode_text(texts, cfg, corrupt=None, rng=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    name = cfg["features"]["text_model"]; pool = cfg["features"]["text_pooling"]
    tok = AutoTokenizer.from_pretrained(name); mdl = AutoModel.from_pretrained(name).eval()
    if corrupt and corrupt[0] == "text":
        texts = [C.token_dropout(t, corrupt[2], rng) for t in texts]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            b = tok(list(map(str, texts[i:i + 32])), padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = mdl(**b).last_hidden_state
            if pool == "first": v = h[:, 0]
            else:
                m = b["attention_mask"].unsqueeze(-1).float(); v = (h * m).sum(1) / m.sum(1)
            out.append(v.numpy())
    return np.concatenate(out).astype(np.float32), np.ones(len(texts), bool)


def _wave(video_path, sr):
    from scipy.io import wavfile
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-ac", "1", "-ar", str(sr), "-vn", wav], check=True)
        rate, data = wavfile.read(wav)
    return data.astype(np.float32) / 32768.0, rate


def encode_audio(paths, cfg, corrupt=None, rng=None):
    import opensmile
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)
    X = np.zeros((len(paths), DIMS["audio"]), np.float32); ok = np.ones(len(paths), bool)
    for i, p in enumerate(paths):
        try:
            w, sr = _wave(p, cfg["features"]["audio_sr"])
            if corrupt and corrupt[0] == "audio": w = C.add_noise_snr(w, corrupt[2], rng)
            X[i] = smile.process_signal(w, sr).values[0]
        except Exception as e:                       # log and flag; never invent a vector
            ok[i] = False; print("audio failure", p, repr(e))
    return X, ok


def encode_video(paths, cfg, corrupt=None, rng=None):
    import cv2, torch, torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity(); net.eval()
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    nf = cfg["features"]["frames_per_clip"]
    X = np.zeros((len(paths), DIMS["video"]), np.float32); ok = np.ones(len(paths), bool)
    for i, p in enumerate(paths):
        try:
            cap = cv2.VideoCapture(p); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []
            for j in np.linspace(0, max(n - 1, 0), nf).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(j)); r, f = cap.read()
                if not r: continue
                if corrupt and corrupt[0] == "video":
                    f = C.blur_frame(f, corrupt[2]) if corrupt[1] == "blur" else C.downscale_frame(f, int(corrupt[2]))
                f = cv2.cvtColor(cv2.resize(f, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frames.append(((f - mean) / std).transpose(2, 0, 1))
            cap.release()
            if not frames: raise RuntimeError("no frames read")
            with torch.no_grad(): X[i] = net(torch.tensor(np.stack(frames))).mean(0).numpy()
        except Exception as e:
            ok[i] = False; print("video failure", p, repr(e))
    return X, ok


def _encode(mod, df, paths, cfg, corrupt, rng, lo, hi):
    if mod == "text": return encode_text(df["text"].tolist()[lo:hi], cfg, corrupt, rng)
    if mod == "audio": return encode_audio(paths[lo:hi], cfg, corrupt, rng)
    return encode_video(paths[lo:hi], cfg, corrupt, rng)


def run(cfg, split, modality, condition=None, corrupt=None, seed=0, chunk=None):
    """Extract in chunks. Each finished chunk is saved atomically, so after a crash or shutdown just run the same
    command again: finished chunks are skipped. Failed items are flagged ok=False and listed, never invented."""
    import time
    df = load_split_csv(os.path.join(cfg["data"]["root"], cfg["data"]["csv"][split]))
    corrupt = C.parse_corruption(corrupt) if corrupt else None
    paths = [media_path(cfg, split, u, "video") for u in df["id"]]
    chunk = int(chunk or cfg["features"].get("chunk_size", 500)); n = len(df); nch = (n + chunk - 1) // chunk
    d = parts_dir(cfg, condition)
    for mod in (MODALITIES if modality == "all" else [modality]):
        xs, oks = [], []
        for c, lo in enumerate(range(0, n, chunk)):
            hi = min(lo + chunk, n); f = f"{d}/{split}_{mod}_chunk{c:04d}.npz"
            if os.path.exists(f):
                z = np.load(f); xs.append(z["X"]); oks.append(z["ok"]); print(f"{split}/{mod} chunk {c + 1}/{nch}: already done, skipped"); continue
            t0 = time.time(); rng = np.random.default_rng(seed * 100003 + c)   # per-chunk rng: resuming reproduces corruption
            X, ok = _encode(mod, df, paths, cfg, corrupt, rng, lo, hi)
            tmp = f[:-4] + ".tmp.npz"; np.savez_compressed(tmp, X=X, ok=ok); os.replace(tmp, f)
            xs.append(X); oks.append(ok)
            print(f"{split}/{mod} chunk {c + 1}/{nch}: {hi - lo} items in {time.time() - t0:.0f} s, {int((~ok).sum())} failed")
        X, ok = np.concatenate(xs), np.concatenate(oks)
        np.save(f"{d}/{split}_{mod}.npy", X); np.save(f"{d}/{split}_{mod}_ok.npy", ok)
        failed = df["id"][~ok].tolist()
        print(f"{split}/{mod}: {int(ok.sum())}/{len(ok)} ok, dim {X.shape[1]}; failed ids (first 20): {failed[:20]}")


def assemble(cfg, split, condition=None):
    """Merge parts into <split>.npz (clean) or <split>__<condition>.npz; a condition file inherits unaffected
    modalities from the clean parts."""
    df = load_split_csv(os.path.join(cfg["data"]["root"], cfg["data"]["csv"][split]))
    feats, oks = {}, []
    for mod in MODALITIES:
        d = parts_dir(cfg, condition); f = f"{d}/{split}_{mod}.npy"
        if not os.path.exists(f): d = parts_dir(cfg, None); f = f"{d}/{split}_{mod}.npy"
        feats[mod] = np.load(f); oks.append(np.load(f.replace(".npy", "_ok.npy")))
    name = f"{split}__{condition}.npz" if condition else f"{split}.npz"
    save_features(os.path.join(cfg["paths"]["features"], name), df["id"].tolist(), df["dialogue"].values, feats,
                  df["emotion"].values, df["sentiment"].values, np.stack(oks, 1))
    print("wrote", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="extract", choices=["extract", "assemble"])
    ap.add_argument("--config", default="configs/experiment.yaml"); ap.add_argument("--split", required=True)
    ap.add_argument("--modality", default="all"); ap.add_argument("--condition"); ap.add_argument("--corrupt")
    ap.add_argument("--chunk", type=int)
    a = ap.parse_args(); cfg = load_config(a.config)
    if a.cmd == "assemble": assemble(cfg, a.split, a.condition)
    else: run(cfg, a.split, a.modality, a.condition, a.corrupt, chunk=a.chunk)
