from methods.common import ce_step
from models.resnet_cifar import build_model


def build(cfg):
    return build_model("vanilla")


def make_step_fn(cfg):
    return ce_step
