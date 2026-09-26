
import argparse
import os
import shutil
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.pacs_protocol import build_source_splits, build_datasets, build_loaders, set_seed, save_json
from shared.pacs import PACS_CLASSES
from task3.models.backbone import build_backbone, FEATURE_DIM
from task3.models.classifier_head import build_classifier
from task3.methods import erm, dan_dg, sam
from task3.evaluation.domain_metrics import evaluate_all_methods
from task3.evaluation.source_domain_separability import compute_source_domain_separability
from task3.evaluation.sharpness import compute_sharpness
from task3.controlled_study import run_controlled_study
from task2.evaluation.plotting import plot_training_curves

N_CLASSES = len(PACS_CLASSES)

REQUIRED_FILES = [
    "comparison_table.json", "comparison_table.md", "domain_gap_analysis.json",
    "source_domain_separability.json", "sharpness_proxy.json",
    "controlled_study_dan_dg_sweep.json", "controlled_study_dan_dg_sweep.md",
    "figures/dan_dg_curves.png", "figures/sam_curves.png",
    "figures/controlled_study_dan_dg_sweep.png",
    "dan_dg_backbone.pt", "dan_dg_classifier.pt",
    "sam_backbone.pt", "sam_classifier.pt",
]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(pacs_root, base_config_path, shared_checkpoint_dir, no_clean, controlled_study):
    cfg = load_config(base_config_path)
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[run_all] WARNING: CUDA not available, falling back to CPU (will be slow).")
    results_dir = cfg["results_dir"]
    fig_dir = os.path.join(results_dir, "figures")

    if not no_clean and os.path.exists(results_dir):
        print(f"[run_all] wiping stale {results_dir}/ ...")
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    set_seed(cfg["seed"])
    splits = build_source_splits(pacs_root, cfg["splits_path"], val_frac=cfg["val_fraction"], seed=cfg["seed"])
    train_ds, val_ds, _tgt_train_ds, sketch_eval_ds = build_datasets(pacs_root, splits)
    source_loaders, _target_loader, sketch_eval_loader, val_loaders = build_loaders(
        train_ds, val_ds, _tgt_train_ds, sketch_eval_ds,
        source_batch_per_domain=cfg["source_batch_per_domain"],
        target_batch_size=cfg["target_batch_size"], num_workers=cfg["num_workers"])

    steps_per_epoch = max(len(ds) // cfg["source_batch_per_domain"] for ds in train_ds.values())
    print(f"[run_all] steps_per_epoch = {steps_per_epoch}")

    common_kwargs = dict(max_epochs=cfg["max_epochs"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                          patience=cfg["patience"], device=device, seed=cfg["seed"])

    models_by_method = {}

    print("\n=== Loading frozen Source-only (ERM) checkpoint from Task 2 ===")
    backbone, classifier = erm.load_erm_checkpoint(shared_checkpoint_dir, device=device)
    models_by_method["erm"] = (backbone, classifier)

    print("\n=== Training DAN-DG (lambda_DG=1) ===")
    set_seed(cfg["seed"])
    backbone = build_backbone(device=device)
    classifier = build_classifier(FEATURE_DIM, N_CLASSES, device=device)
    result = dan_dg.train(backbone, classifier, source_loaders, val_loaders,
                           steps_per_epoch=steps_per_epoch, results_dir=results_dir,
                           lambda_dg=1.0, **common_kwargs)
    plot_training_curves(result["history"], "dan_dg", fig_dir)
    models_by_method["dan_dg"] = (backbone, classifier)

    print("\n=== Training SAM (rho=0.05) ===")
    set_seed(cfg["seed"])
    backbone = build_backbone(device=device)
    classifier = build_classifier(FEATURE_DIM, N_CLASSES, device=device)
    result = sam.train(backbone, classifier, source_loaders, val_loaders,
                        steps_per_epoch=steps_per_epoch, results_dir=results_dir,
                        rho=0.05, **common_kwargs)
    plot_training_curves(result["history"], "sam", fig_dir)
    models_by_method["sam"] = (backbone, classifier)

    print("\n=== Final evaluation on Sketch + source-domain separability + sharpness ===")
    evaluate_all_methods(models_by_method, val_loaders, sketch_eval_loader, results_dir, device=device)

    separability = {method: compute_source_domain_separability(bb, val_loaders, seed=cfg["seed"], device=device)
                    for method, (bb, _clf) in models_by_method.items()}
    save_json(separability, os.path.join(results_dir, "source_domain_separability.json"))

    sharpness = {method: compute_sharpness(bb, clf, val_loaders, epsilon=0.05, seed=cfg["seed"], device=device)
                 for method, (bb, clf) in models_by_method.items()}
    save_json(sharpness, os.path.join(results_dir, "sharpness_proxy.json"))

    if controlled_study:
        print(f"\n=== Controlled study: {controlled_study} ===")
        run_controlled_study(source_loaders, val_loaders, steps_per_epoch, results_dir, cfg,
                              device=device, study=controlled_study)

    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(results_dir, f))]
    print("\n" + "=" * 60)
    if missing:
        print("[run_all] INCOMPLETE -- the following required files are MISSING:")
        for f in missing:
            print(f"    - {f}")
        raise RuntimeError(
            f"{len(missing)} required evidence file(s) missing under {results_dir}/. "
            f"Fix the above and rerun.")
    print(f"\n[run_all] All output written to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pacs_root", type=str, required=True)
    parser.add_argument("--shared_checkpoint_dir", type=str, default="shared/checkpoints",
                         help="Where Task 2's frozen source_only_backbone.pt / source_only_classifier.pt live.")
    parser.add_argument("--base_config", type=str, default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "base.yaml"))
    parser.add_argument("--no_clean", action="store_true",
                         help="Skip wiping results/ first (not recommended).")
    parser.add_argument("--controlled_study", type=str, default="dan_dg",
                         choices=["dan_dg", "sam", "none"],
                         help="Which of the two bounded studies to run (spec: choose ONE).")
    args = parser.parse_args()
    main(args.pacs_root, args.base_config, args.shared_checkpoint_dir, args.no_clean,
         None if args.controlled_study == "none" else args.controlled_study)
