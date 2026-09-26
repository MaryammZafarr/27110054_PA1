
import torch


def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    sq = (x ** 2).sum(dim=1, keepdim=True)
    return sq + sq.t() - 2 * x @ x.t()


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, bandwidth_muls=(0.5, 1.0, 2.0)) -> torch.Tensor:
  
    n_x = x.shape[0]
    combined = torch.cat([x, y], dim=0)
    sq_dists = _pairwise_sq_dists(combined)

    off_diag_mask = ~torch.eye(sq_dists.shape[0], dtype=torch.bool, device=x.device)
    median_sq_dist = sq_dists[off_diag_mask].median().clamp_min(1e-8)

    kernel_sum = torch.zeros_like(sq_dists)
    for mul in bandwidth_muls:
        bandwidth = mul * median_sq_dist
        kernel_sum = kernel_sum + torch.exp(-sq_dists / (2 * bandwidth))

    k_xx = kernel_sum[:n_x, :n_x]
    k_yy = kernel_sum[n_x:, n_x:]
    k_xy = kernel_sum[:n_x, n_x:]
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def pairwise_domain_mmd(features_by_domain: dict) -> torch.Tensor:
    """Average MMD^2 over all unordered pairs of domains (used by DAN-DG)."""
    domains = list(features_by_domain.keys())
    total, n_pairs = 0.0, 0
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            total = total + rbf_mmd2(features_by_domain[domains[i]], features_by_domain[domains[j]])
            n_pairs += 1
    return total / n_pairs
