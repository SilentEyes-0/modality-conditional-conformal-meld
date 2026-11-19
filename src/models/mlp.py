"""Two-head fusion MLP in pure NumPy (manual backprop, Adam). Input 1,371 -> 256 -> 128 -> {7, 3}.
Chosen so the whole training/calibration stack is deterministic, testable on CPU and GPU-free."""
import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


class FusionMLP:
    def __init__(self, d_in, h1=256, h2=128, k_e=7, k_s=3, dropout=0.3, seed=0):
        rng = np.random.default_rng(seed)
        he = lambda i, o: rng.normal(0, np.sqrt(2.0 / i), size=(i, o)).astype(np.float32)
        sm = lambda i, o: rng.normal(0, np.sqrt(1.0 / i), size=(i, o)).astype(np.float32)
        self.p = {"W1": he(d_in, h1), "b1": np.zeros(h1, np.float32), "W2": he(h1, h2), "b2": np.zeros(h2, np.float32),
                  "We": sm(h2, k_e), "be": np.zeros(k_e, np.float32), "Ws": sm(h2, k_s), "bs": np.zeros(k_s, np.float32)}
        self.dropout = dropout

    def forward(self, X, train=False, rng=None):
        p = self.p
        z1 = X @ p["W1"] + p["b1"]; a1 = np.maximum(z1, 0)
        m1 = m2 = None
        if train and self.dropout > 0:
            m1 = (rng.random(a1.shape) >= self.dropout).astype(np.float32) / (1 - self.dropout); a1 = a1 * m1
        z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(z2, 0)
        if train and self.dropout > 0:
            m2 = (rng.random(a2.shape) >= self.dropout).astype(np.float32) / (1 - self.dropout); a2 = a2 * m2
        le = a2 @ p["We"] + p["be"]; ls = a2 @ p["Ws"] + p["bs"]
        return le, ls, (X, z1, a1, m1, z2, a2, m2)

    def backward(self, cache, ge, gs):
        p = self.p; X, z1, a1, m1, z2, a2, m2 = cache
        g = {"We": a2.T @ ge, "be": ge.sum(0), "Ws": a2.T @ gs, "bs": gs.sum(0)}
        da2 = ge @ p["We"].T + gs @ p["Ws"].T
        if m2 is not None: da2 = da2 * m2
        dz2 = da2 * (z2 > 0)
        g["W2"] = a1.T @ dz2; g["b2"] = dz2.sum(0)
        da1 = dz2 @ p["W2"].T
        if m1 is not None: da1 = da1 * m1
        dz1 = da1 * (z1 > 0)
        g["W1"] = X.T @ dz1; g["b1"] = dz1.sum(0)
        return g

    def logits(self, X, batch=2048):
        outs_e, outs_s = [], []
        for i in range(0, len(X), batch):
            le, ls, _ = self.forward(X[i:i + batch], train=False)
            outs_e.append(le); outs_s.append(ls)
        return np.concatenate(outs_e), np.concatenate(outs_s)

    def get_params(self): return {k: v.copy() for k, v in self.p.items()}
    def set_params(self, params): self.p = {k: v.copy() for k, v in params.items()}
    def n_params(self): return int(sum(v.size for v in self.p.values()))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k] + (self.wd * params[k] if k.startswith("W") else 0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t); vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(params[k].dtype)


def loss_and_grads(model, X, ye, ys, lam, w_e=None, train=True, rng=None):
    le, ls, cache = model.forward(X, train=train, rng=rng)
    n = len(X)
    pe, ps = softmax(le), softmax(ls)
    we = np.ones(n, np.float32) if w_e is None else w_e[ye].astype(np.float32)
    nll_e = -(np.log(pe[np.arange(n), ye] + 1e-12) * we).mean()
    nll_s = -np.log(ps[np.arange(n), ys] + 1e-12).mean()
    ge = pe.copy(); ge[np.arange(n), ye] -= 1; ge = ge * we[:, None] / n
    gs = ps.copy(); gs[np.arange(n), ys] -= 1; gs = lam * gs / n
    return nll_e + lam * nll_s, model.backward(cache, ge, gs)
