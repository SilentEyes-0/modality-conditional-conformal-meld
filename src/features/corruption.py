"""Corruption operators applied to RAW signals before feature extraction (levels are fixed in the analysis plan)."""
import numpy as np


def add_noise_snr(wave, snr_db, rng):
    """Additive white Gaussian noise at a target signal-to-noise ratio (dB)."""
    p = np.mean(wave ** 2) + 1e-12
    noise = rng.normal(0, np.sqrt(p / (10 ** (snr_db / 10))), size=wave.shape)
    return (wave + noise).astype(wave.dtype)


def blur_frame(frame, ksize):
    import cv2
    k = int(ksize) | 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def downscale_frame(frame, factor):
    import cv2
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // factor), max(1, h // factor)))
    return cv2.resize(small, (w, h))


def token_dropout(text, p, rng):
    words = text.split()
    keep = [w for w in words if rng.random() >= p]
    return " ".join(keep) if keep else (words[0] if words else text)


def parse_corruption(spec):
    """'audio:snr=10' | 'video:blur=15' | 'video:down=4' | 'text:drop=0.2' -> (modality, kind, value)"""
    mod, rest = spec.split(":"); kind, val = rest.split("=")
    return mod, kind, float(val)
