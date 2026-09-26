
import numpy as np


def per_class_accuracy(preds, labels, n_classes=None):
    from shared.pacs import PACS_CLASSES
    n_classes = n_classes or len(PACS_CLASSES)
    out = {}
    for c in range(n_classes):
        mask = labels == c
        out[PACS_CLASSES[c]] = float("nan") if mask.sum() == 0 else \
            float((preds[mask] == labels[mask]).float().mean())
    return out


def build_comparison_table(results_by_method: dict) -> list:
    """Required Evidence #1: one row per method, target-accuracy CHANGE
    relative to source_only, plus domain separability."""
    baseline_target_acc = results_by_method["source_only"]["target_accuracy"]
    rows = []
    for name, r in results_by_method.items():
        rows.append({
            "method": name,
            **{f"val_acc_{d}": r["source_val_by_domain"][d]["accuracy"] for d in r["source_val_by_domain"]},
            **{f"val_f1_{d}": r["source_val_by_domain"][d]["macro_f1"] for d in r["source_val_by_domain"]},
            "mean_source_accuracy": r["mean_source_accuracy"],
            "mean_source_macro_f1": r["mean_source_macro_f1"],
            "target_accuracy": r["target_accuracy"],
            "target_macro_f1": r["target_macro_f1"],
            "target_accuracy_change_vs_source_only": r["target_accuracy"] - baseline_target_acc,
            "domain_separability_accuracy": r["domain_separability"]["domain_separability_accuracy"],
        })
    return rows


def comparison_table_to_markdown(rows: list) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def per_class_deltas_vs_source_only(results_by_method: dict) -> dict:
    """Required Evidence #3 (part 1): per-class accuracy change vs Source-only."""
    baseline = results_by_method["source_only"]["per_class_target_accuracy"]
    deltas = {}
    for name, r in results_by_method.items():
        if name == "source_only":
            continue
        deltas[name] = {cls: r["per_class_target_accuracy"][cls] - baseline[cls] for cls in baseline}
    return deltas


def top_confusions_for_class(cm, class_names, class_name: str, k: int = 3):
    """Top-k classes `class_name` is confused WITH (excluding itself), read
    off row `class_name` of the confusion matrix (rows=true, cols=pred)."""
    cm = np.array(cm)
    idx = class_names.index(class_name)
    row = cm[idx].copy()
    row[idx] = -1
    top_idx = np.argsort(row)[::-1][:k]
    return [(class_names[i], int(row[i])) for i in top_idx if row[i] > 0]