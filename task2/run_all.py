
import argparse
import os
import shutil
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.pacs_protocol import build_source_splits, build_datasets, build_loaders, set_seed
from shared.pacs import PACS_CLASSES
from task2.models.backbone import build_backbone, FEATURE_DIM
from task2.models.classifier_head import build_classifier
from task2.models.domain_discriminator import DomainDiscriminator
from task2.methods import source_only, dan, dann, cdan
from task2.evaluate_final import run_common_evaluation
from task2.controlled_study import run_controlled_study
from task2.evaluation.plotting import plot_training_curves

N_CLASSES = len(PACS_CLASSES)

REQUIRED_FILES = [
    "comparison_table.json", "comparison_table.md", "full_evaluation_results.json",
    "domain_gap_analysis.json", "per_class_failure_analysis.json",
    "controlled_study_dan_lambda_sweep.json", "controlled_study_dan_lambda_sweep.md",
    "figures/source_only_curves.png", "figures/dan_curves.png",
    "figures/dann_curves.png", "figures/cdan_curves.png",
    "figures/controlled_study_dan_lambda_sweep.png",
    "source_only_backbone.pt", "source_only_classifier.pt",
    "dan_backbone.pt", "dan_classifier.pt",
    "dann_backbone.pt", "dann_classifier.pt", "dann_aux.pt",
    "cdan_backbone.pt", "cdan_classifier.pt", "cdan_aux.pt",
]


def load_config(base_config_path):
    with open(base_config_path) as f:
        return yaml.safe_load(f)


def main(pacs_root: str, base_config_path: str, no_clean: bool):
    cfg = load_config(base_config_path)
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[run_all] WARNING: CUDA not available, falling back to CPU (will be slow).")
    results_dir = cfg["results_dir"]
    fig_dir = os.path.join(results_dir, "figures")

    if cfg.get("cudnn_benchmark", True) and device == "cuda":
        torch.backends.cudnn.benchmark = True

    if not no_clean and os.path.exists(results_dir):
        print(f"[run_all] wiping stale {results_dir}/ so this run's evidence is self-consistent...")
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    set_seed(cfg["seed"])
    print("[run_all] building datasets/loaders...")
    splits = build_source_splits(pacs_root, cfg["splits_path"], val_frac=cfg["val_fraction"], seed=cfg["seed"])
    train_ds, val_ds, tgt_train_ds, tgt_eval_ds = build_datasets(pacs_root, splits)
    source_loaders, target_loader, target_eval_loader, val_loaders = build_loaders(
        train_ds, val_ds, tgt_train_ds, tgt_eval_ds,
        source_batch_per_domain=cfg["source_batch_per_domain"],
        target_batch_size=cfg["target_batch_size"], num_workers=cfg["num_workers"])

    steps_per_epoch = max(len(ds) // cfg["source_batch_per_domain"] for ds in train_ds.values())
    print(f"[run_all] steps_per_epoch = {steps_per_epoch}")

    common_kwargs = dict(max_epochs=cfg["max_epochs"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                          patience=cfg["patience"], device=device, seed=cfg["seed"],
                          use_amp=cfg.get("use_amp", False))

    models_by_method = {}

    print("\n=== Training Source-only ERM ===")
    set_seed(cfg["seed"])
    backbone = build_backbone(device=device)
    classifier = build_classifier(FEATURE_DIM, N_CLASSES, device=device)
    result = source_only.train(backbone, classifier, source_loaders, val_loaders,
                                steps_per_epoch=steps_per_epoch, results_dir=results_dir, **common_kwargs)
    plot_training_curves(result["history"], "source_only", fig_dir)
    models_by_method["source_only"] = (backbone, classifier)
    shared_ckpt_dir = cfg["shared_checkpoint_dir"]
    os.makedirs(shared_ckpt_dir, exist_ok=True)
    torch.save(backbone.state_dict(), os.path.join(shared_ckpt_dir, "source_only_backbone.pt"))
    torch.save(classifier.state_dict(), os.path.join(shared_ckpt_dir, "source_only_classifier.pt"))
    print(f"[run_all] Source-only checkpoint copied -> {shared_ckpt_dir} (frozen Task 3 baseline)")

    print("\n=== Training DAN ===")
    set_seed(cfg["seed"])
    backbone = build_backbone(device=device)
    classifier = build_classifier(FEATURE_DIM, N_CLASSES, device=device)
    result = dan.train(backbone, classifier, source_loaders, target_loader, val_loaders,
                        steps_per_epoch=steps_per_epoch, results_dir=results_dir,
                        lambda_mmd=1.0, **common_kwargs)
    plot_training_curves(result["history"], "dan", fig_dir)
    models_by_method["dan"] = (backbone, classifier)

    print("\n=== Training DANN ===")
    set_seed(cfg["seed"])
    backbone = build_backbone(device=device)
    classifier = build_classifier(FEATURE_DIM, N_CLASSES, device=device)
    discriminator = DomainDiscriminator(in_dim=FEATURE_DIM).to(device)
    result = dann.train(backbone, classifier, discriminator, source_loaders, target_loader,
                         val_loaders, steps_per_epoch=steps_per_epoch, results_dir=results_dir,
                         max_alpha=1, **common_kwargs)
    plot_training_curves(result["history"], "dann", fig_dir)
    models_by_method["dann"] = (backbone, classifier)

    print("\n=== Training CDAN ===")
    set_seed(cfg["seed"])
    backbone = build_backbone(device=device)
    classifier = build_classifier(FEATURE_DIM, N_CLASSES, device=device)
    discriminator = DomainDiscriminator(in_dim=FEATURE_DIM * N_CLASSES).to(device)
    result = cdan.train(backbone, classifier, discriminator, source_loaders, target_loader,
                         val_loaders, steps_per_epoch=steps_per_epoch, results_dir=results_dir,
                         max_alpha=1.0, **common_kwargs)
    plot_training_curves(result["history"], "cdan", fig_dir)
    models_by_method["cdan"] = (backbone, classifier)

    print("\n=== Common evaluation + domain separability + per-class analysis ===")
    run_common_evaluation(models_by_method, val_loaders, target_eval_loader, results_dir, device=device)

    print("\n=== Controlled study: DAN lambda_MMD sweep ===")
    run_controlled_study(source_loaders, target_loader, target_eval_loader, val_loaders,
                          steps_per_epoch, results_dir, cfg, device=device)

    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(results_dir, f))]
    print("\n" + "=" * 60)
    if missing:
        print("[run_all] INCOMPLETE — the following required files are MISSING:")
        for f in missing:
            print(f"    - {f}")
        raise RuntimeError(
            f"{len(missing)} required evidence file(s) missing under {results_dir}/. "
            f"Fix the above and rerun.")
    else:
        print(f"\n[run_all] All output written to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pacs_root", type=str, required=True)
    parser.add_argument("--base_config", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "base.yaml"))
    parser.add_argument("--no_clean", action="store_true",
                         help="Skip wiping results/ first (not recommended).")
    args = parser.parse_args()
    main(args.pacs_root, args.base_config, args.no_clean)
