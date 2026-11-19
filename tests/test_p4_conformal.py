import numpy as np
from src.constants import PATTERNS, FULL
from src.conformal import core as cp
from src.calibration.temperature import fit_temperature
from src.models.mlp import log_softmax

def test_quantile_rule_edges():
    assert cp.conformal_quantile([0.3, 0.1, 0.2], 0.5) == 0.2          # n=3, k=ceil(4*0.5)=2
    assert np.isinf(cp.conformal_quantile([0.1] * 8, 0.1))              # n=8: k=9>n -> full set
    assert cp.conformal_quantile(list(range(1, 10)), 0.1) == 9          # n=9 is the smallest non-trivial n for alpha=.1
    assert np.isinf(cp.conformal_quantile([], 0.1))


def test_global111_equals_conditional_pattern_111_design_A():
    rng = np.random.default_rng(2)
    sA = {m: rng.random(500) for m in PATTERNS}
    assert cp.global_111(sA, 0.1) == cp.conformal_quantile(sA[FULL], 0.1)


def test_mondrian_fallback_levels_use_counts_only():
    rng = np.random.default_rng(3)
    masks = np.array([7] * 200 + [6] * 5 + [5] * 5 + [3] * 5 + [4] * 40 + [2] * 40 + [1] * 5)
    scores = rng.random(len(masks))
    tab = cp.mondrian(scores, masks, 0.1, n_min=30)
    assert tab[7][1] == 0 and tab[4][1] == 0                             # enough data
    assert tab[6][1] == 2 and tab[5][1] == 2 and tab[3][1] == 2          # 2-modality group has 15 < 30 -> global
    assert tab[1][1] == 1                                                # 1-modality group has 85 >= 30 -> level 1
    assert tab[6][0] == cp.conformal_quantile(scores, 0.1)               # level 2 equals the global threshold


def test_marginal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(4); covs = []
    for _ in range(300):
        n = 200
        cal = rng.random(n); test = rng.random(2000)
        q = cp.conformal_quantile(cal, 0.1); covs.append((test <= q).mean())
    assert np.mean(covs) >= 0.9 - 0.005 and np.mean(covs) <= 0.9 + 1 / 201 + 0.01


def test_mondrian_pattern_coverage_with_heterogeneous_groups():
    """Validity is a statement about AVERAGE coverage over calibration draws, so average over repeats."""
    rng = np.random.default_rng(5); cm = {m: [] for m in PATTERNS}; cg = {m: [] for m in PATTERNS}
    sig = {7: 0.5, 6: 1.0, 5: 1.0, 3: 1.5, 4: 1.5, 2: 2.0, 1: 2.5}
    for _ in range(100):
        masks = rng.choice(PATTERNS, size=3000)
        sc = np.abs(rng.normal(0, np.array([sig[m] for m in masks])))
        tm = rng.choice(PATTERNS, size=6000); ts = np.abs(rng.normal(0, np.array([sig[m] for m in tm])))
        qg = cp.conformal_quantile(sc, 0.1); tab = cp.mondrian(sc, masks, 0.1, 30)
        for m in PATTERNS:
            i = tm == m; cm[m].append((ts[i] <= tab[m][0]).mean()); cg[m].append((ts[i] <= qg).mean())
    dev_m = max(abs(np.mean(v) - 0.9) for v in cm.values()); dev_g = max(abs(np.mean(v) - 0.9) for v in cg.values())
    assert dev_m < 0.01          # Mondrian: every pattern near the target on average
    assert dev_g > 0.15          # global threshold: some pattern is far from the target


def test_prediction_sets_threshold():
    p = np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
    assert cp.prediction_sets(p, 0.65).tolist() == [[True, False, False], [True, True, False]]   # include y iff p(y) >= 0.35
    assert cp.prediction_sets(p, 0.85).tolist() == [[True, True, False], [True, True, True]]     # include y iff p(y) >= 0.15
    assert cp.prediction_sets(p, np.inf).all()                                                   # +inf threshold -> full label set
    assert cp.prediction_sets(p, np.array([0.65, 0.85])).tolist() == [[True, False, False], [True, True, True]]


def test_coarser_taxonomies():
    rng = np.random.default_rng(7)
    masks = np.array([7] * 100 + [6] * 20 + [5] * 20 + [3] * 20 + [4] * 20 + [2] * 20 + [1] * 20)
    sc = rng.random(len(masks))
    card = cp.mondrian_taxonomy(sc, masks, 0.1, 30, "cardinality")
    assert card[6][1] == 0 and card[6][2] == 60 and card[1][2] == 60       # 2-modality and 1-modality groups pooled
    text = cp.mondrian_taxonomy(sc, masks, 0.1, 30, "text")
    assert text[7][2] == 160 and text[2][2] == 60                           # text present / absent group sizes
    tiny = cp.mondrian_taxonomy(sc, masks, 0.1, 500, "exact")
    assert all(v[1] == 2 for v in tiny.values())                            # nothing reaches n_min -> global


def test_temperature_reduces_nll_for_overconfident_logits():
    rng = np.random.default_rng(2); n = 4000
    y = rng.integers(0, 7, n); z = rng.normal(size=(n, 7)); z[np.arange(n), y] += 1.0
    z_over = 5.0 * z                                                    # artificially over-confident
    T = fit_temperature(z_over, y)
    nll = lambda L: -log_softmax(L)[np.arange(n), y].mean()
    assert T > 2.0 and nll(z_over / T) < nll(z_over)
