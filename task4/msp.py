import numpy as np


def msp_score(logits: np.ndarray) -> np.ndarray:
    """u_MSP = 1 - max_k softmax_k(z)   (Hendrycks & Gimpel 2017). float64 for numerical safety.
    NOTE: saturates at exactly 0 for very confident samples -> ties; evaluation/analysis reports how many."""
    z = logits.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return 1.0 - p.max(1)
