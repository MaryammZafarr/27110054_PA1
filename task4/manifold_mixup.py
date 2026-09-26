import torch


def sample_lambda(alpha, batch, device, per_sample=False):
    """lambda ~ Beta(alpha, alpha). Paper Alg.2 samples ONE lambda per mini-batch (default here)."""
    dist = torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(alpha)))
    lam = dist.sample((batch,) if per_sample else ()).to(device)
    return lam.view(-1, 1, 1, 1) if per_sample else lam


def pick_partners(y, generator=None):
    """For every i choose a random j in the same mini-batch with y_j != y_i (shuffle + mask same-class pairs).

    Returns (partner_index, valid_mask). Rows with no different-class partner (essentially impossible with
    batch 64 / 10 classes) are marked invalid and dropped from the loss.
    """
    diff = (y[:, None] != y[None, :])
    valid = diff.any(1)
    w = diff.float() + (~valid).float()[:, None]      # avoid all-zero rows in multinomial
    j = torch.multinomial(w, 1, generator=generator).squeeze(1)
    return j, valid


def manifold_mixup(h, y, alpha=2.0, per_sample=False, generator=None):
    """h = phi_pre(x) (activations after layer2). Returns (h_mix, valid_mask, partner_idx, lam)."""
    j, valid = pick_partners(y, generator)
    lam = sample_lambda(alpha, h.size(0), h.device, per_sample)
    h_mix = lam * h + (1 - lam) * h[j]
    return h_mix, valid, j, lam
