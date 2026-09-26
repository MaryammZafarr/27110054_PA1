import numpy as np


def placeholder_score(logits_all: np.ndarray, num_known: int = 10) -> np.ndarray:
    """PROSER placeholder-based unknownness (reference-code rule, paper Sec. 4.3).

    u_PH(x) = max_c d_c(x) - max_k z_k(x)      (strongest dummy response minus strongest known response)

    Paper: logits become [z, d_max + bias]; x is UNKNOWN iff the (calibrated) dummy is the arg-max, i.e.
    u_PH + bias > 0. The bias is picked so 95% of CIFAR-10 VALIDATION data are accepted, which is identical to
    thresholding u_PH at its 95th validation percentile (tau = -bias) -- exactly the manual's protocol.
    """
    z = logits_all.astype(np.float64)
    return z[:, num_known:].max(1) - z[:, :num_known].max(1)
