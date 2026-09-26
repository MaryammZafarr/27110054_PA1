
import torch
import torch.nn as nn

CE = nn.CrossEntropyLoss()


def _fixed_batch(val_loaders: dict, n_per_domain: int = 32, seed: int = 6304):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for domain, loader in sorted(val_loaders.items()):
        ds = loader.dataset
        n = len(ds)
        perm = torch.randperm(n, generator=g)[:n_per_domain]
        for idx in perm.tolist():
            x, y = ds[idx]
            xs.append(x)
            ys.append(y)
    return torch.stack(xs), torch.tensor(ys)


def compute_sharpness(backbone, classifier, val_loaders: dict, epsilon: float = 0.05,
                       seed: int = 6304, device: str = "cuda") -> dict:
    backbone.eval()
    classifier.eval()
    x, y = _fixed_batch(val_loaders, seed=seed)
    x, y = x.to(device), y.to(device)

    params = [p for p in list(backbone.parameters()) + list(classifier.parameters()) if p.requires_grad]
    for p in params:
        p.grad = None

    logits = classifier(backbone(x))
    loss_before = CE(logits, y)
    loss_before.backward()

    grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
    norms = [g.norm(2) for g in grads if g is not None]
    grad_norm = torch.norm(torch.stack(norms), 2) if norms else torch.tensor(0.0)
    scale = epsilon / (grad_norm + 1e-12)

    original = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, g in zip(params, grads):
            if g is not None:
                p.add_(g * scale)

        logits_after = classifier(backbone(x))
        loss_after = CE(logits_after, y)

        for p, orig in zip(params, original):
            p.copy_(orig)

    for p in params:
        p.grad = None

    delta_sharp = float(loss_after.item() - loss_before.item())
    return {"loss_before": float(loss_before.item()), "loss_after": float(loss_after.item()),
            "delta_sharp": delta_sharp, "epsilon": epsilon}
