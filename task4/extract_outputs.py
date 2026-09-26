
import argparse
import datetime
import os

import numpy as np
import torch

from data.cifar10 import get_eval_loader
from methods.manifold_mixup import pick_partners, sample_lambda
from models.resnet_cifar import build_model
from utils import NUM_KNOWN, SEED, ensure_dir, get_device, load_config, load_json, parse_overrides, save_json, set_seed, sha256_file


def load_model(name, ckpt_dir, device):
    ck = torch.load(os.path.join(ckpt_dir, name, "best.pt"), map_location=device)
    m = ck["meta"]
    model = build_model(m["method"], num_dummy=m.get("num_dummy", 5), num_points=m.get("num_points", 1))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def run(model, loader, device):
    Z, F, Y = [], [], []
    for x, y, _ in loader:
        z, f = model(x.to(device), return_feat=True)
        Z.append(z.float().cpu()); F.append(f.float().cpu()); Y.append(y)
    return torch.cat(Z).numpy(), torch.cat(F).numpy(), torch.cat(Y).numpy()


@torch.no_grad()
def run_val_mixup(model, loader, device, seed=SEED, alpha=2.0):
    """Proxy unknowns built from CIFAR-10 VALIDATION data only: manifold mixup after layer2 between different
    classes (lambda ~ Beta(2,2), per-sample). Used only to study whether interpolation ~ real unknowns."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Z, F = [], []
    for x, y, _ in loader:
        h = model.forward_pre(x.to(device))
        j, valid = pick_partners(y, g)                       # CPU-side partner choice (deterministic)
        j = j.to(device)
        lam = sample_lambda(alpha, h.size(0), device, per_sample=True)
        z, f = model.forward_post(lam * h + (1 - lam) * h[j], return_feat=True)
        keep = valid.to(device)
        Z.append(z[keep].float().cpu()); F.append(f[keep].float().cpu())
    return torch.cat(Z).numpy(), torch.cat(F).numpy()


def freeze(names, ckpt_dir, manifest_path, refreeze):
    new = {}
    for n in names:
        p = os.path.join(ckpt_dir, n, "best.pt")
        assert os.path.exists(p), f"missing checkpoint {p} -- train {n} first"
        ck = torch.load(p, map_location="cpu")
        new[n] = dict(path=p, sha256=sha256_file(p), best_epoch=ck["epoch"], val_acc=ck["val_acc"])
    old = load_json(manifest_path) if os.path.exists(manifest_path) else {}
    for n, rec in new.items():
        if n in old.get("models", {}) and old["models"][n]["sha256"] != rec["sha256"] and not refreeze:
            raise RuntimeError(f"checkpoint '{n}' changed AFTER unknowns were evaluated (hash mismatch). "
                               f"This violates the protocol. Use --refreeze only if you re-ran everything honestly.")
    manifest = dict(models={**old.get("models", {}), **new}, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    refrozen=bool(refreeze) or old.get("refrozen", False),
                    note="Checkpoints hashed BEFORE any CIFAR-100 image was loaded.")
    save_json(manifest, manifest_path)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vanilla", "gcsc", "proser"])
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--refreeze", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="overrides (debug_n=.., unk_per_class=..) for sanity checks")
    a = ap.parse_args()
    ov = parse_overrides(a.set)
    cfg = load_config("configs/vanilla.yaml", ov)
    set_seed(SEED)
    device = get_device()
    cache = ensure_dir(cfg.get("cache_dir", "cache"))
    manifest_path = os.path.join(cfg.get("results_dir", "results"), "freeze_manifest.json")

    # ---- 1. freeze ----
    freeze(a.models, a.ckpt_dir, manifest_path, a.refreeze)
    print("checkpoints frozen ->", manifest_path)

    # ---- 2. known-class outputs (NO CIFAR-100 loaded yet) ----
    models = {}
    for n in a.models:
        model, ck = load_model(n, a.ckpt_dir, device)
        models[n] = model
        out = {}
        for split in ("train", "val", "test"):
            z, f, y = run(model, get_eval_loader(cfg, split), device)
            out[f"{split}_logits"], out[f"{split}_feats"], out[f"{split}_labels"] = z, f, y
        out["valmix_logits"], out["valmix_feats"] = run_val_mixup(model, get_eval_loader(cfg, "val"), device)
        np.savez_compressed(os.path.join(cache, f"{n}_known.npz"), **out)
        acc = (out["test_logits"][:, :NUM_KNOWN].argmax(1) == out["test_labels"]).mean()
        print(f"[{n}] best epoch {ck['epoch']} val_acc {ck['val_acc']:.4f} | CIFAR-10 test acc (10 known logits) {acc:.4f}")

    # ---- 3. unknowns: imported only now ----
    from data.cifar100_unknowns import build_unknown_set, unknown_loader
    unk = build_unknown_set(cfg["data_root"], per_class_limit=cfg.get("unk_per_class"))
    meta = {k: v for k, v in unk.items() if k != "x"}                     # cache = features/logits/metadata only
    np.savez_compressed(os.path.join(cache, "unknown_meta.npz"), **meta)
    print(f"unknowns: near={int((unk['group'] == 'near').sum())} far={int((unk['group'] == 'far').sum())}")
    for n, model in models.items():
        z, f, _ = run(model, unknown_loader(unk, cfg.get("num_workers", 2)), device)
        np.savez_compressed(os.path.join(cache, f"{n}_unknown.npz"), logits=z, feats=f)
        print(f"[{n}] unknown outputs cached ({len(z)} images)")
    print("done. cache/ holds features+logits only; checkpoints are hashed in", manifest_path)


if __name__ == "__main__":
    main()
