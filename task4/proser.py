
import torch
import torch.nn.functional as F

from methods.manifold_mixup import manifold_mixup
from models.resnet_cifar import build_model
from utils import NUM_KNOWN


def build(cfg):
    """PROSER = selected Vanilla checkpoint + C randomly initialised dummy classifiers (extra rows of fc)."""
    K, C = NUM_KNOWN, cfg["num_dummy"]
    ck = torch.load(cfg["init_from"], map_location="cpu")
    torch.manual_seed(cfg["seed"])                       # reproducible dummy initialisation
    model = build_model("proser", num_dummy=C)          # fc: Linear(512, K + C), PyTorch default init
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) == {"fc.weight", "fc.bias"} and not unexpected, (missing, unexpected)
    with torch.no_grad():                                # keep the trained known-class classifiers
        model.fc.weight[:K] = ck["model"]["fc.weight"]
        model.fc.bias[:K] = ck["model"]["fc.bias"]
    return model


def _augment_with_dummy(z, K):
    """[z_known, max dummy]  -> (B, K+1)"""
    return torch.cat([z[:, :K], z[:, K:].max(1, keepdim=True).values], dim=1)


def make_step_fn(cfg):
    K, beta, gamma = NUM_KNOWN, cfg["beta"], cfg["gamma"]
    alpha, per_sample = cfg.get("mixup_alpha", 2.0), cfg.get("mix_per_sample", False)

    def step(model, x, y):
        h = x.size(0) // 2
        x1, y1, x2, y2 = x[:h], y[:h], x[h:], y[h:]

        z1 = model(x1).float()
        aug1 = _augment_with_dummy(z1, K)                                 
        l_ce = F.cross_entropy(aug1, y1)                                   # keep true class largest
        masked = aug1.masked_fill(F.one_hot(y1, K + 1).bool(), -1e9)       # remove ground-truth logit
        dummy_target = torch.full_like(y1, K)
        l_cp = F.cross_entropy(masked, dummy_target)                       # dummy must win among the rest

        # ---- data placeholders (second half) ----
        h2 = model.forward_pre(x2)                                         # phi_pre(x): after layer2
        h_mix, valid, _, _ = manifold_mixup(h2, y2, alpha, per_sample)
        z2 = model.forward_post(h_mix).float()
        aug2 = _augment_with_dummy(z2, K)
        l_dp = F.cross_entropy(aug2[valid], torch.full_like(y2[valid], K))

        loss = l_ce + beta * l_cp + gamma * l_dp
        stats = dict(l_ce=l_ce.item(), l_cp=l_cp.item(), l_dp=l_dp.item(),
                     acc=(z1[:, :K].argmax(1) == y1).float().mean().item(),
                     mix_dummy_wins=(aug2.argmax(1) == K).float().mean().item())
        return loss, stats

    return step
