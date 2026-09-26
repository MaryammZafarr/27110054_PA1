"""
Train a single method given its config (merged over configs/base.yaml).
Callable standalone:
    python task2/train.py --pacs_root /path/to/PACS --config configs/dan.yaml
or imported and called directly by run_all.py to reuse already-built
loaders (avoids rebuilding the dataset 4 times in one process).
"""
import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.pacs_protocol import set_seed
from task2.models.backbone import build_backbone, FEATURE_DIM
from task2.models.classifier_head import build_classifier
from task2.models.domain_discriminator import DomainDiscriminator
from shared.pacs import PACS_CLASSES

N_CLASSES = len(PACS_CLASSES)


def load_config(base_path, method_path):
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    with open(method_path) as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def train_one_method(cfg, source_loaders, target_loader, val_loaders, steps_per_epoch, results_dir):
    """Builds a fresh backbone/classifier(/discriminator), trains the given
    method, and returns (backbone, classifier, train_result)."""
    device = cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"
    method_name = cfg["method_name"]

    set_seed(cfg["seed"])  # identical init + sampling order across all methods
    backbone = build_backbone(device=device)
    classifier = build_classifier(FEATURE_DIM, N_CLASSES, device=device)

    common_kwargs = dict(
        max_epochs=cfg["max_epochs"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        patience=cfg["patience"], device=device, seed=cfg["seed"], use_amp=cfg.get("use_amp", False))

    if method_name == "source_only":
        from task2.methods import source_only
        result = source_only.train(backbone, classifier, source_loaders, val_loaders,
                                    steps_per_epoch=steps_per_epoch, results_dir=results_dir,
                                    **common_kwargs)
    elif method_name == "dan":
        from task2.methods import dan
        result = dan.train(backbone, classifier, source_loaders, target_loader, val_loaders,
                            steps_per_epoch=steps_per_epoch, results_dir=results_dir,
                            lambda_mmd=cfg["lambda_mmd"], **common_kwargs)
    elif method_name in ("dann", "cdan"):
        module = __import__(f"task2.methods.{method_name}", fromlist=["train"])
        disc_in_dim = FEATURE_DIM if method_name == "dann" else FEATURE_DIM * N_CLASSES
        discriminator = DomainDiscriminator(in_dim=disc_in_dim).to(device)
        result = module.train(backbone, classifier, discriminator, source_loaders, target_loader,
                               val_loaders, steps_per_epoch=steps_per_epoch, results_dir=results_dir,
                               max_alpha=cfg["max_alpha"], **common_kwargs)
    else:
        raise ValueError(f"Unknown method_name in config: {method_name}")

    return backbone, classifier, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pacs_root", required=True)
    parser.add_argument("--config", required=True, help="e.g. configs/dan.yaml")
    parser.add_argument("--base_config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.base_config, args.config)
    results_dir = cfg["results_dir"]
    splits_path = cfg["splits_path"]

    if cfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    from shared.pacs_protocol import build_source_splits, build_datasets, build_loaders
    splits = build_source_splits(args.pacs_root, splits_path, val_frac=cfg["val_fraction"], seed=cfg["seed"])
    train_ds, val_ds, tgt_train_ds, tgt_eval_ds = build_datasets(args.pacs_root, splits)
    source_loaders, target_loader, target_eval_loader, val_loaders = build_loaders(
        train_ds, val_ds, tgt_train_ds, tgt_eval_ds,
        source_batch_per_domain=cfg["source_batch_per_domain"],
        target_batch_size=cfg["target_batch_size"], num_workers=cfg["num_workers"])

    steps_per_epoch = max(len(ds) // cfg["source_batch_per_domain"] for ds in train_ds.values())
    print(f"steps_per_epoch = {steps_per_epoch}")

    backbone, classifier, result = train_one_method(
        cfg, source_loaders, target_loader, val_loaders, steps_per_epoch, results_dir)

    from task2.evaluation.plotting import plot_training_curves
    plot_training_curves(result["history"], cfg["method_name"], os.path.join(results_dir, "figures"))

    if cfg["method_name"] == "source_only":
        shared_ckpt_dir = cfg["shared_checkpoint_dir"]
        os.makedirs(shared_ckpt_dir, exist_ok=True)
        torch.save(backbone.state_dict(), os.path.join(shared_ckpt_dir, "source_only_backbone.pt"))
        torch.save(classifier.state_dict(), os.path.join(shared_ckpt_dir, "source_only_classifier.pt"))
        print(f"[train] copied Source-only checkpoint -> {shared_ckpt_dir} (frozen Task 3 ERM baseline)")


if __name__ == "__main__":
    main()