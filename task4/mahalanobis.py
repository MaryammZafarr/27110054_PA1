import numpy as np


class MahalanobisScorer:
    """u_Mah(x) = min_c (f-mu_c)^T Sigma^-1 (f-mu_c)

    Class means mu_c and ONE shared DIAGONAL covariance Sigma are estimated from UNAUGMENTED CIFAR-10 TRAIN
    features (train portion only). DECISION: Sigma is the pooled WITHIN-class variance
    (mean over all samples of (f - mu_{y})^2, per dimension) + 1e-6 on every diagonal entry -- the standard
    shared-covariance estimator used for Mahalanobis OOD scoring (not the total variance, which would
    include the between-class spread).
    """

    def __init__(self, eps=1e-6):
        self.eps = eps

    def fit(self, feats: np.ndarray, labels: np.ndarray, num_classes=10):
        f = feats.astype(np.float64)
        self.mu = np.stack([f[labels == c].mean(0) for c in range(num_classes)])
        centred = f - self.mu[labels]
        self.var = (centred ** 2).mean(0) + self.eps
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        f = feats.astype(np.float64)
        d = np.stack([(((f - m) ** 2) / self.var).sum(1) for m in self.mu], 1)  # (N, C)
        return d.min(1)
