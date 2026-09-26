
import torch
import torch.nn.functional as F

from models.resnet_cifar import build_model


def build(cfg):
    return build_model("rpl", num_points=cfg.get("num_points", 1))


def make_step_fn(cfg):
    lam_o, temp = cfg.get("lambda_open", 0.1), cfg.get("temperature", 1.0)

    def step(model, x, y):
        logits, f = model(x, return_feat=True)
        logits = logits.float()
        l_c = F.cross_entropy(logits / temp, y)
        d_own = model.fc.euclid_to_own_points(f.float(), y)
        l_o = ((d_own - model.fc.radius) ** 2).mean()
        loss = l_c + lam_o * l_o
        return loss, dict(l_c=l_c.item(), l_o=l_o.item(), acc=(logits.argmax(1) == y).float().mean().item(),
                          R=model.fc.radius.item())

    return step
