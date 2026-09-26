
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.pacs_protocol import save_json
from shared.pacs import PACS_CLASSES
from task2.evaluation.metrics import evaluate_method
from task2.evaluation.domain_separability import domain_separability_score
from task2.evaluation.class_analysis import (
    build_comparison_table, comparison_table_to_markdown,
    per_class_deltas_vs_source_only, top_confusions_for_class)

METHODS = ["source_only", "dan", "dann", "cdan"]


def run_common_evaluation(models_by_method: dict, val_loaders, target_eval_loader,
                           results_dir: str, device: str = "cuda"):
    results_by_method = {}
    for name, (backbone, classifier) in models_by_method.items():
        eval_result, _, _ = evaluate_method(backbone, classifier, val_loaders,
                                             target_eval_loader, name, device=device)
        sep_result = domain_separability_score(backbone, val_loaders, target_eval_loader, device=device)
        eval_result["domain_separability"] = sep_result
        results_by_method[name] = eval_result

    comparison_table = build_comparison_table(results_by_method)
    save_json(comparison_table, os.path.join(results_dir, "comparison_table.json"))
    with open(os.path.join(results_dir, "comparison_table.md"), "w") as f:
        f.write(comparison_table_to_markdown(comparison_table))
    save_json(results_by_method, os.path.join(results_dir, "full_evaluation_results.json"))

    so = results_by_method["source_only"]
    worst_classes = sorted(so["per_class_target_accuracy"].items(), key=lambda kv: kv[1])[:3]
    domain_gap_analysis = {
        "mean_source_accuracy": so["mean_source_accuracy"],
        "target_accuracy": so["target_accuracy"],
        "domain_gap_accuracy": so["mean_source_accuracy"] - so["target_accuracy"],
        "mean_source_macro_f1": so["mean_source_macro_f1"],
        "target_macro_f1": so["target_macro_f1"],
        "domain_gap_macro_f1": so["mean_source_macro_f1"] - so["target_macro_f1"],
        "per_class_target_accuracy": so["per_class_target_accuracy"],
        "worst_classes": worst_classes,
        "worst_class_top_confusions": {
            cls: top_confusions_for_class(so["confusion_matrix"], PACS_CLASSES, cls)
            for cls, _ in worst_classes
        },
    }
    save_json(domain_gap_analysis, os.path.join(results_dir, "domain_gap_analysis.json"))

    deltas = per_class_deltas_vs_source_only(results_by_method)
    failure_analysis = {}
    for method, class_deltas in deltas.items():
        sorted_classes = sorted(class_deltas.items(), key=lambda kv: kv[1])
        cm = results_by_method[method]["confusion_matrix"]
        failure_analysis[method] = {
            "per_class_accuracy_delta_vs_source_only": class_deltas,
            "most_degraded_classes": [
                {"class": c, "delta": d, "top_confusions": top_confusions_for_class(cm, PACS_CLASSES, c)}
                for c, d in sorted_classes[:2]
            ],
            "most_improved_classes": [
                {"class": c, "delta": d, "top_confusions": top_confusions_for_class(cm, PACS_CLASSES, c)}
                for c, d in sorted_classes[-2:]
            ],
        }
    save_json(failure_analysis, os.path.join(results_dir, "per_class_failure_analysis.json"))

    return results_by_method, comparison_table, domain_gap_analysis, failure_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pacs_root", required=True)
    parser.add_argument("--base_config", default="configs/base.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.base_config) as f:
        cfg = yaml.safe_load(f)
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    results_dir = cfg["results_dir"]

    from shared.pacs_protocol import build_source_splits, build_datasets, build_loaders
    from task2.models.backbone import build_backbone, FEATURE_DIM
    from task2.models.classifier_head import build_classifier

    splits = build_source_splits(args.pacs_root, cfg["splits_path"], val_frac=cfg["val_fraction"], seed=cfg["seed"])
    train_ds, val_ds, tgt_train_ds, tgt_eval_ds = build_datasets(args.pacs_root, splits)
    _, _, target_eval_loader, val_loaders = build_loaders(
        train_ds, val_ds, tgt_train_ds, tgt_eval_ds, num_workers=cfg["num_workers"])

    models_by_method = {}
    for name in METHODS:
        backbone = build_backbone(device=device)
        classifier = build_classifier(FEATURE_DIM, len(PACS_CLASSES), device=device)
        backbone.load_state_dict(torch.load(os.path.join(results_dir, f"{name}_backbone.pt"), map_location=device))
        classifier.load_state_dict(torch.load(os.path.join(results_dir, f"{name}_classifier.pt"), map_location=device))
        models_by_method[name] = (backbone, classifier)

    run_common_evaluation(models_by_method, val_loaders, target_eval_loader, results_dir, device=device)
    print(f"[evaluate_final] wrote comparison_table.{{json,md}}, full_evaluation_results.json, "
          f"domain_gap_analysis.json, per_class_failure_analysis.json -> {results_dir}")


if __name__ == "__main__":
    main()