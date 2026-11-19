import numpy as np, pytest
from src.constants import PATTERNS, D_IN
from src.masking import build_input, mask_bits, Standardizer, sample_masks
from src.models.mlp import FusionMLP, loss_and_grads, softmax
from src.models.train import train_fusion, rho_probs

def test_input_dimension_and_zeroing():
    n = 5; rng = np.random.default_rng(0)
    f = {"text": rng.normal(size=(n, 768)), "audio": rng.normal(size=(n, 88)), "video": rng.normal(size=(n, 512))}
    X = build_input(f, np.array([7, 6, 5, 3, 4]))
    assert X.shape == (n, D_IN) == (n, 1371)
    assert np.all(X[1, 768 + 88:768 + 88 + 512] == 0)                    # pattern 110: video zeroed
    assert np.all(X[4, 768:768 + 88 + 512] == 0)                          # pattern 100: audio+video zeroed
    assert X[3, -3:].tolist() == [0, 1, 1] and X[0, -3:].tolist() == [1, 1, 1]
    assert np.all(mask_bits(np.array([4]))[0] == [1, 0, 0])


def test_standardizer_uses_fit_stats_only():
    rng = np.random.default_rng(1)
    fit = {m: rng.normal(5, 2, size=(200, d)) for m, d in [("text", 768), ("audio", 88), ("video", 512)]}
    s = Standardizer().fit(fit); out = s.transform(fit)
    assert abs(out["audio"].mean()) < 1e-3
    other = {m: v + 100 for m, v in fit.items()}
    assert s.transform(other)["text"].mean() > 10                       # not re-fitted on new data


def test_sample_masks_never_empty():
    m = sample_masks(5000, np.random.default_rng(0))
    assert set(np.unique(m)) == set(PATTERNS) and 0 not in m


def test_gradient_check_manual_backprop():
    rng = np.random.default_rng(0)
    net = FusionMLP(6, 5, 4, 7, 3, dropout=0.0, seed=1)
    net.p = {k: v.astype(np.float64) for k, v in net.p.items()}
    X = rng.normal(size=(8, 6)); ye = rng.integers(0, 7, 8); ys = rng.integers(0, 3, 8)
    _, g = loss_and_grads(net, X, ye, ys, lam=0.7, train=False)
    eps = 1e-6; worst = 0
    for k in net.p:
        flat = net.p[k].reshape(-1)
        for j in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[j]; flat[j] = old + eps; lp, _ = loss_and_grads(net, X, ye, ys, 0.7, train=False)
            flat[j] = old - eps; lm, _ = loss_and_grads(net, X, ye, ys, 0.7, train=False); flat[j] = old
            num = (lp - lm) / (2 * eps); ana = g[k].reshape(-1)[j]
            worst = max(worst, abs(num - ana) / max(1e-7, abs(num) + abs(ana)))
    assert worst < 1e-4


def test_overfits_tiny_separable_problem():
    from src.models.mlp import Adam
    rng = np.random.default_rng(1); n = 60
    ye = rng.integers(0, 7, n); ys = rng.integers(0, 3, n)
    X = np.eye(7)[ye] @ rng.normal(size=(7, 20)) + 0.05 * rng.normal(size=(n, 20))
    net = FusionMLP(20, 16, 8, dropout=0.0, seed=0); opt = Adam(net.p, lr=0.01)
    first, _ = loss_and_grads(net, X, ye, ys, 1.0, train=False)
    for _ in range(200):
        _, g = loss_and_grads(net, X, ye, ys, 1.0, train=False); opt.step(net.p, g)
    last, _ = loss_and_grads(net, X, ye, ys, 1.0, train=False)
    assert last < 0.5 * first


def _toy(n, rng):
    em = rng.integers(0, 7, n); sent = rng.integers(0, 3, n)
    protos = {m: np.random.default_rng(k).normal(size=(7, d)) for k, (m, d) in enumerate([("text", 768), ("audio", 88), ("video", 512)])}
    feats = {m: (1.5 * protos[m][em] + rng.normal(size=(n, protos[m].shape[1]))).astype(np.float32) for m in protos}
    return {"feats": feats, "emotion": em, "sentiment": sent}


def test_train_fusion_runs_stops_early_and_modes_differ():
    rng = np.random.default_rng(0); fit, sel = _toy(300, rng), _toy(100, rng)
    t = dict(h1=32, h2=16, dropout=0.2, lr=0.003, weight_decay=1e-4, batch_size=64, epochs=12, patience=3, lambda_sentiment=1.0, rho_full=0.4)
    m1, h1 = train_fusion(fit, sel, "dropout", t, 0); m2, h2 = train_fusion(fit, sel, "full", t, 0)
    assert h1["best_val_nll"] < h1["val_nll"][0] and 1 <= h1["epochs_run"] <= 12
    assert abs(sum(rho_probs(0.4).values()) - 1) < 1e-12
    m3, _ = train_fusion(fit, sel, ("single", 4), t, 0)
    X = build_input(sel["feats"], np.full(100, 4)); assert m3.logits(X)[0].shape == (100, 7)
    m1b, h1b = train_fusion(fit, sel, "dropout", t, 0); assert h1b["val_nll"] == h1["val_nll"]      # deterministic
