"""One scalar temperature per task, fitted by minimising NLL on the temperature-fit subset (after training is frozen)."""
import numpy as np
from scipy.optimize import minimize_scalar
from ..models.mlp import log_softmax


def fit_temperature(logits, y, bounds=(0.05, 20.0)):
    def nll(logT):
        lp = log_softmax(logits / np.exp(logT))
        return -lp[np.arange(len(y)), y].mean()
    r = minimize_scalar(nll, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(r.x))


def apply_temperature(logits, T):
    from ..models.mlp import softmax
    return softmax(logits / T)
