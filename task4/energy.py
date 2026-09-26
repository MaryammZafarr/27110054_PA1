import numpy as np
from scipy.special import logsumexp


def energy_score(logits: np.ndarray) -> np.ndarray:
    """u_Energy = -log sum_k exp(z_k)   (Liu et al. 2020, temperature T=1; uses ALL logits)."""
    return -logsumexp(logits.astype(np.float64), axis=1)
