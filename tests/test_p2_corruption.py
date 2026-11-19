import numpy as np
from src.features import corruption as C
from src.features.timing import extrapolate, fmt

def test_snr_corruption_hits_target():
    rng = np.random.default_rng(0); w = rng.normal(size=200000).astype(np.float32)
    for snr in (20, 10, 0):
        n = C.add_noise_snr(w, snr, rng) - w
        est = 10 * np.log10(np.mean(w ** 2) / np.mean(n ** 2)); assert abs(est - snr) < 0.3


def test_token_dropout_keeps_something():
    rng = np.random.default_rng(0)
    assert len(C.token_dropout("a b c d e f", 1.0, rng).split()) >= 1
    assert C.parse_corruption("audio:snr=10") == ("audio", "snr", 10.0)


def test_timing_extrapolation_removes_one_off_loading_cost():
    # model load = 10 s, per item = 0.2 s: t(50) = 10 + 10, t(100) = 10 + 20
    est, per = extrapolate(20.0, 30.0, 50, 13708)
    assert abs(per - 0.2) < 1e-12 and abs(est - 0.2 * 13708) < 1e-9
    assert "h" in fmt(est)
