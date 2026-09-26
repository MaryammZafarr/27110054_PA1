import numpy as np


def mls_score(logits: np.ndarray) -> np.ndarray:
    """u_MLS = -max_k z_k  (Maximum Logit Score; keeps absolute logit magnitude -- Vaze et al. 2022)."""
    return -logits.astype(np.float64).max(1)
