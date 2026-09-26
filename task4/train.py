
import argparse
import importlib
import sys

import torch

from data.cifar10 import get_train_val_loaders
from methods.common import fit
from models.resnet_cifar import count_params
from utils import SEED, get_device, load_config, parse_overrides, set_seed

METHODS = {"vanilla": "methods.vanilla", "gcsc": "methods.gcsc", "proser": "methods.proser", "rpl": "methods.rpl"}


def assert_no_cifar100():
    leaked = [m for m in sys.modules if "cifar100" in m.lower()]
    assert not leaked, f"PROTOCOL VIOLATION: CIFAR-100 module imported during training: {leaked}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. epochs=2 debug_n=256")
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    assert cfg["seed"] == SEED, "manual requires seed 6304"
    assert_no_cifar100()
    set_seed(cfg["seed"])
    device = get_device()
    mod = importlib.import_module(METHODS[cfg["method"]])
    model = mod.build(cfg).to(device).to(memory_format=torch.channels_last)
    step_fn = mod.make_step_fn(cfg)
    train_loader, val_loader = get_train_val_loaders(cfg)
    print(f"[{cfg['name']}] device={device} params={count_params(model):,} train_batches/epoch={len(train_loader)} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} aug={cfg['augmentation']}")
    best_acc, best_ep = fit(cfg, model, step_fn, train_loader, val_loader, cfg["ckpt_dir"], device, resume=a.resume)
    assert_no_cifar100()
    print(f"[{cfg['name']}] DONE  best val acc = {best_acc:.4f} at epoch {best_ep}  -> {cfg['ckpt_dir']}/best.pt")


if __name__ == "__main__":
    main()
