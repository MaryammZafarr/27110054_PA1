import torch.nn as nn


def build_classifier(in_dim: int, n_classes: int, device: str = "cuda") -> nn.Module:
    return nn.Linear(in_dim, n_classes).to(device)