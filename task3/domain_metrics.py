
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from shared.pacs import PACS_CLASSES
from shared.pacs_protocol import evaluate_on_loader, save_json


@torch.no_grad()
def _per_class_sketch_eval(backbone, classifier, sketch_eval_loader, device: str = "cuda"):
    """Returns per-class accuracy dict + confusion matrix (rows=true, cols=pred)
    over PACS_CLASSES order, computed directly since evaluate_on_loader only
    returns aggregate accuracy/macro_f1."""
    backbone.eval()
    classifier.eval()
    all_preds, all_labels = [], []
    for x, y in sketch_eval_loader:
        x = x.to(device, non_blocking=True)
        logits = classifier(backbone(x))
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    cm = sk_confusion_matrix(labels, preds, labels=list(range(len(PACS_CLASSES))))
    per_class_acc = {}
    for i, cls in enumerate(PACS_CLASSES):
        n_true = cm[i].sum()
        per_class_acc[cls] = float(cm[i, i] / n_true) if n_true > 0 else None
    return per_class_acc, cm


def _top_confusions_for_class(cm: np.ndarray, cls_idx: int, top_k: int = 2):
    row = cm[cls_idx].copy()
    n_true = row.sum()
    row[cls_idx] = 0  # exclude correct predictions
    top_idx = np.argsort(row)[::-1][:top_k]
    return [{"predicted_as": PACS_CLASSES[j], "count": int(row[j]),
             "fraction_of_class": float(row[j] / n_true) if n_true > 0 else None}
            for j in top_idx if row[j] > 0]


def evaluate_all_methods(models_by_method: dict, val_loaders: dict, sketch_eval_loader,
                          results_dir: str, device: str = "cuda"):
    """models_by_method: {"erm": (backbone, classifier), "dan_dg": (...), "sam": (...)}"""
    table = {}
    per_class_by_method = {}
    confusion_by_method = {}

    for method, (backbone, classifier) in models_by_method.items():
        source_metrics = {d: evaluate_on_loader(backbone, classifier, l, device=device)
                           for d, l in val_loaders.items()}
        accs = [m["accuracy"] for m in source_metrics.values()]
        f1s = [m["macro_f1"] for m in source_metrics.values()]
        sketch_metrics = evaluate_on_loader(backbone, classifier, sketch_eval_loader, device=device)

        per_class_acc, cm = _per_class_sketch_eval(backbone, classifier, sketch_eval_loader, device=device)
        per_class_by_method[method] = per_class_acc
        confusion_by_method[method] = cm

        table[method] = {
            "source_validation": source_metrics,
            "mean_source_accuracy": sum(accs) / len(accs),
            "mean_source_macro_f1": sum(f1s) / len(f1s),
            "worst_source_accuracy": min(accs),
            "worst_source_macro_f1": min(f1s),
            "sketch_accuracy": sketch_metrics["accuracy"],
            "sketch_macro_f1": sketch_metrics["macro_f1"],
            "sketch_per_class_accuracy": per_class_acc,
        }

    erm_sketch_acc = table["erm"]["sketch_accuracy"]
    for method in table:
        table[method]["sketch_accuracy_change_vs_erm"] = table[method]["sketch_accuracy"] - erm_sketch_acc

    save_json(table, os.path.join(results_dir, "domain_gap_analysis.json"))
    save_json(table, os.path.join(results_dir, "comparison_table.json"))
    save_json(table, os.path.join(results_dir, "full_evaluation_results.json"))
    _write_markdown_table(table, os.path.join(results_dir, "comparison_table.md"))

    # --- Required Evidence #3: per-class Sketch changes vs ERM --- #
    erm_per_class = per_class_by_method["erm"]
    failure_analysis = {}
    for method in models_by_method:
        if method == "erm":
            continue
        cm = confusion_by_method[method]
        deltas = {}
        for i, cls in enumerate(PACS_CLASSES):
            erm_acc = erm_per_class[cls]
            m_acc = per_class_by_method[method][cls]
            deltas[cls] = (None if erm_acc is None or m_acc is None
                            else m_acc - erm_acc)
        valid = {c: d for c, d in deltas.items() if d is not None}
        sorted_classes = sorted(valid.items(), key=lambda kv: kv[1])

        failure_analysis[method] = {
            "sketch_per_class_accuracy": per_class_by_method[method],
            "erm_sketch_per_class_accuracy": erm_per_class,
            "per_class_accuracy_delta_vs_erm": deltas,
            "most_degraded_classes": [
                {"class": c, "delta": d,
                 "top_confusions": _top_confusions_for_class(cm, PACS_CLASSES.index(c))}
                for c, d in sorted_classes[:2]
            ],
            "most_improved_classes": [
                {"class": c, "delta": d,
                 "top_confusions": _top_confusions_for_class(cm, PACS_CLASSES.index(c))}
                for c, d in sorted_classes[-2:]
            ],
        }
    save_json(failure_analysis, os.path.join(results_dir, "per_class_failure_analysis.json"))

    return table, failure_analysis


def _write_markdown_table(table: dict, path: str):
    domains = list(next(iter(table.values()))["source_validation"].keys())
    header = ["method"] + [f"{d}_acc" for d in domains] + [f"{d}_f1" for d in domains] + \
             ["mean_src_acc", "mean_src_f1", "worst_src_acc", "worst_src_f1",
              "sketch_acc", "sketch_f1", "sketch_acc_delta_vs_erm"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for method, row in table.items():
        vals = [method]
        vals += [f"{row['source_validation'][d]['accuracy']:.4f}" for d in domains]
        vals += [f"{row['source_validation'][d]['macro_f1']:.4f}" for d in domains]
        vals += [f"{row['mean_source_accuracy']:.4f}", f"{row['mean_source_macro_f1']:.4f}",
                 f"{row['worst_source_accuracy']:.4f}", f"{row['worst_source_macro_f1']:.4f}",
                 f"{row['sketch_accuracy']:.4f}", f"{row['sketch_macro_f1']:.4f}",
                 f"{row['sketch_accuracy_change_vs_erm']:+.4f}"]
        lines.append("| " + " | ".join(vals) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines))