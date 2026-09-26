
import math

import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
    def forward(self, x):
        return self.net(x)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


def grl_alpha_schedule(p: float, max_alpha: float = 1.0) -> float:
    """
    Standard DANN schedule: alpha(p) = 2/(1+exp(-10p)) - 1, p in [0,1] is
    training progress. `max_alpha` rescales the [0,1]-ranged schedule while
    keeping its SHAPE identical — used only by the controlled alignment-
    strength study.
    """
    base = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
    return max_alpha * base