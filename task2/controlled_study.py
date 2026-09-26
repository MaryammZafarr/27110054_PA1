
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from shared.pacs_protocol import save_json
from task2.models.backbone import build_backbone, FEATURE_DIM
from task2.models.classifier_head import build_classifier
from task2.methods.dan import train as train_dan
from task2.evaluation.metrics import evaluate_method
from task2.evaluation.domain_separability import domain_separability_score
from task2.evaluation.plotting import plot_training_curves, plot_controlled_study
from task2.evaluation.class_analysis import comparison_table_to_markdown
from shared.pacs import PACS_CLASSES

LAMBDA_SWEEP = [0.1, 1.0, 10.0]


def run_controlled_study(source_loaders, target_loader, target_eval_loader, val_loaders,
                          steps_per_epoch, results_dir, cfg, device="cuda"):
    sweep_results = []
    fig_dir = os.path.join(results_dir, "figures")
    n_classes = len(PACS_CLASSES)

    common_kwargs = dict(max_epochs=cfg["max_epochs"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                          patience=cfg["patience"], device=device, seed=cfg["seed"],
                          use_amp=cfg.get("use_amp", False))

    for lam in LAMBDA_SWEEP:
        from shared.pacs_protocol import set_seed
        set_seed(cfg["seed"])
        backbone = build_backbone(device=device)
        classifier = build_classifier(FEATURE_DIM, n_classes, device=device)
        method_name = f"dan_lambda{lam}"

        train_result = train_dan(
            backbone, classifier, source_loaders, target_loader, val_loaders,
            steps_per_epoch=steps_per_epoch, results_dir=results_dir,
            lambda_mmd=lam, method_name=method_name, **common_kwargs)
        plot_training_curves(train_result["history"], method_name, fig_dir)

        eval_result, _, _ = evaluate_method(backbone, classifier, val_loaders,
                                             target_eval_loader, method_name, device=device)
        sep_result = domain_separability_score(backbone, val_loaders, target_eval_loader, device=device)

        sweep_results.append({
            "lambda_mmd": lam,
            "mean_source_accuracy": eval_result["mean_source_accuracy"],
            "mean_source_macro_f1": eval_result["mean_source_macro_f1"],
            "target_accuracy": eval_result["target_accuracy"],
            "target_macro_f1": eval_result["target_macro_f1"],
            "domain_separability_accuracy": sep_result["domain_separability_accuracy"],
        })
        print(f"[controlled study] lambda_mmd={lam}: {sweep_results[-1]}")

    save_json(sweep_results, os.path.join(results_dir, "controlled_study_dan_lambda_sweep.json"))
    with open(os.path.join(results_dir, "controlled_study_dan_lambda_sweep.md"), "w") as f:
        f.write(comparison_table_to_markdown(sweep_results))
    plot_controlled_study(sweep_results, os.path.join(fig_dir, "controlled_study_dan_lambda_sweep.png"))
    return sweep_results