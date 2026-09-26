
import argparse
import ast
import glob
import os
import sys
import time

import numpy as np
import torch
import torchvision

from utils import SEED, load_config

OK = lambda m: print(f"  [ok] {m}")


def check_env(bench):
    print("1. environment")
    print(f"  torch {torch.__version__} | torchvision {torchvision.__version__} | python {sys.version.split()[0]}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        OK(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GiB")
    else:
        print("CPU")


def check_data(cfg):
    print("2. data + split")
    from data.cifar10 import load_cifar10
    from data.make_splits import load_or_make_split, make_split, verify_split
    x, y = load_cifar10(cfg["data_root"], True); xt, yt = load_cifar10(cfg["data_root"], False)
    assert x.shape == (50000, 32, 32, 3) and xt.shape == (10000, 32, 32, 3), (x.shape, xt.shape)
    assert x.dtype == np.uint8 and set(np.unique(y)) == set(range(10))
    OK("CIFAR-10: 50000 train / 10000 test, uint8 32x32x3, 10 classes")
    path = os.path.join(cfg["cache_dir"], "splits", f"cifar10_split_seed{SEED}.npz")
    tr, va = load_or_make_split(y, path, SEED)
    verify_split(y, tr, va)
    tr2, va2 = make_split(y, SEED)
    assert (tr == tr2).all() and (va == va2).all(), "split is not reproducible from seed 6304"
    OK("stratified 90/10 split (45000/5000, 500 per class), disjoint, reproducible from seed 6304")
    # unknown inventory (counts only)
    from data.cifar100_unknowns import FAR_CLASSES, NEAR_CLASSES, build_unknown_set
    u = build_unknown_set(cfg["data_root"])
    assert (u["group"] == "near").sum() == 800 and (u["group"] == "far").sum() == 800
    OK(f"unknowns: near {NEAR_CLASSES} = 800 imgs; far {FAR_CLASSES} = 800 imgs; groups disjoint")


def check_model():
    print("3. architecture")
    from models.resnet_cifar import build_model, count_params
    m = build_model("vanilla")
    assert m.conv1.kernel_size == (3, 3) and m.conv1.stride == (1, 1), "first conv must be 3x3 stride 1"
    assert not hasattr(m, "maxpool"), "initial max-pool must be removed"
    x = torch.randn(2, 3, 32, 32)
    assert m(x).shape == (2, 10) and m.forward_pre(x).shape == (2, 128, 16, 16)
    OK(f"CIFAR ResNet-18: 3x3/s1 stem, no max-pool, params={count_params(m):,}, layer2 output (128,16,16) -> mixup point OK")
    assert build_model("proser")(x).shape == (2, 15)
    OK("PROSER head: 10 known + 5 dummy outputs")


def check_transforms():
    print("4. augmentation pipelines")
    from data.cifar10 import build_transform
    names = lambda k: [type(t).__name__ for t in build_transform(k).transforms]
    assert names("base") == ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"], names("base")
    assert names("randaugment") == ["RandomCrop", "RandomHorizontalFlip", "RandAugment", "ToTensor", "Normalize"]
    ra = build_transform("randaugment").transforms[2]
    assert ra.num_ops == 2 and ra.magnitude == 9
    assert names("eval") == ["ToTensor", "Normalize"]
    OK("base = crop(pad 4)+flip; GCSC = crop+flip+RandAugment(2, 9) BEFORE ToTensor; eval = no augmentation")


def check_configs():
    print("5. configs vs manual")
    v, g, p, r = (load_config(f"configs/{n}.yaml") for n in ("vanilla", "gcsc", "proser", "rpl"))
    for c in (v, g, p, r):
        assert c["seed"] == SEED and c["momentum"] == 0.9 and c["weight_decay"] == 5e-4 and c["scheduler"] == "cosine"
    assert (v["lr"], v["epochs"], v["batch_size"]) == (0.1, 100, 128)
    diff = {k for k in set(v) | set(g) if v.get(k) != g.get(k)}
    assert diff == {"name", "method", "augmentation", "ckpt_dir"}, f"GCSC differs from Vanilla in {diff}"
    assert g["augmentation"] == "randaugment" and v["augmentation"] == "base"
    OK("GCSC config differs from Vanilla ONLY in augmentation (plus name/method/output dir)")
    assert (p["epochs"], p["lr"], p["batch_size"], p["num_dummy"], p["beta"], p["gamma"], p["mixup_alpha"]) == (50, 1e-3, 128, 5, 1.0, 0.1, 2.0)
    assert p["init_from"].endswith("vanilla/best.pt")
    OK("PROSER: 50 epochs, lr 1e-3, batch 128, 5 dummies, beta=1, gamma=0.1, Beta(2,2), init from selected Vanilla checkpoint")
    if r["grad_clip"]:
        print("  [WARN] RPL uses grad_clip -> report it as a deviation")


def check_no_cifar100_in_training():
    print("6. leakage guard (static scan of training code)")
    files = ["train.py", "utils.py", "data/cifar10.py", "data/make_splits.py", "models/resnet_cifar.py"] + glob.glob("methods/*.py")
    for f in files:
        tree = ast.parse(open(f).read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            for n in names:
                bad = "cifar100_unknowns" in n.lower() or n.lower() in ("cifar100", "load_cifar100_test", "build_unknown_set")
                assert not bad, f"{f}: references CIFAR-100 ({n})"
    OK(f"{len(files)} training-side files never import or reference CIFAR-100")


def bench(cfg, steps):
    print("7. speed benchmark")
    from data.cifar10 import get_train_val_loaders
    from methods.common import ce_step
    from models.resnet_cifar import build_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("vanilla").to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    for aug in ("base", "randaugment"):
        loader, _ = get_train_val_loaders(dict(cfg, augmentation=aug))
        it, t0 = iter(loader), None
        for i in range(steps + 2):
            if i == 2:
                t0 = time.time()
            x, y, _ = next(it)
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
                loss, _ = ce_step(model, x, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        per = (time.time() - t0) / steps
        ep = per * len(loader)
        print(f"  aug={aug:12s}: {per * 1000:.0f} ms/step -> ~{ep:.0f}s/epoch -> 100 epochs ~ {ep * 100 / 3600:.1f} h "
              f"(if this is dominated by data loading, raise num_workers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench_steps", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config("configs/vanilla.yaml")
    check_env(a.bench); check_data(cfg); check_model(); check_transforms(); check_configs(); check_no_cifar100_in_training()
    if a.bench:
        bench(cfg, a.bench_steps)
    print("\nPREFLIGHT PASSED -- now go through README section A (manual checklist) and run sanity_check.py")


if __name__ == "__main__":
    main()
