
import torch
import torch.nn as nn
import torchvision


class RPLHead(nn.Module):


    def __init__(self, feat_dim, num_classes, num_points=1, init_scale=0.1):
        super().__init__()
        self.K, self.M, self.d = num_classes, num_points, feat_dim
        self.points = nn.Parameter(init_scale * torch.randn(num_classes * num_points, feat_dim))
        self.radius = nn.Parameter(torch.zeros(1))  # learnable bound R of the open-space regulariser

    def _terms(self, f):
        f = f.float()
        p = self.points.float()
        d_e = (f.pow(2).sum(1, keepdim=True) - 2 * f @ p.t() + p.pow(2).sum(1)[None]) / self.d
        d_d = f @ p.t()
        return d_e, d_d

    def forward(self, f):
        d_e, d_d = self._terms(f)
        return (d_e - d_d).view(-1, self.K, self.M).mean(2)

    def euclid_to_own_points(self, f, y):
        d_e, _ = self._terms(f)
        d_e = d_e.view(-1, self.K, self.M).mean(2)
        return d_e.gather(1, y[:, None]).squeeze(1)


class ResNet18CIFAR(nn.Module):
    feat_dim = 512

    def __init__(self, num_outputs=10, head="linear", num_points=1):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.bn1, self.relu = base.bn1, base.relu
        # NOTE: no max-pool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if head == "linear":
            self.fc = nn.Linear(512, num_outputs)
        elif head == "rpl":
            self.fc = RPLHead(512, num_outputs, num_points)
        else:
            raise ValueError(head)

    def forward_pre(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.layer2(self.layer1(x))

    def forward_post(self, h, return_feat=False):
        h = self.layer4(self.layer3(h))
        f = torch.flatten(self.avgpool(h), 1)          # penultimate feature f(x) (512-d)
        z = self.fc(f)
        return (z, f) if return_feat else z

    def forward(self, x, return_feat=False):
        return self.forward_post(self.forward_pre(x), return_feat)


def build_model(method: str, num_dummy: int = 5, num_points: int = 1):
    """Factory used by train.py and extract_outputs.py (so checkpoints always rebuild identically)."""
    if method in ("vanilla", "gcsc"):
        return ResNet18CIFAR(10)
    if method == "proser":
        return ResNet18CIFAR(10 + num_dummy)
    if method == "rpl":
        return ResNet18CIFAR(10, head="rpl", num_points=num_points)
    raise ValueError(method)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
