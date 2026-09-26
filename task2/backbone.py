
import torch.nn as nn
import torchvision.models as tvm

FEATURE_DIM = 512


def build_backbone(device: str = "cuda") -> nn.Module:
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1
    net = tvm.resnet18(weights=weights)
    net.fc = nn.Identity()
    return net.to(device)