
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from shared.pacs_protocol import evaluate_on_loader
from shared.pacs import PACS_CLASSES


@torch.no_grad()
def target_predictions(backbone, classifier, target_eval_loader, device="cuda"):
    backbone.eval()
    classifier.eval()
    all_preds, all_labels = [], []
    for x, y in target_eval_loader:
        x = x.to(device, non_blocking=True)
        logits = classifier(backbone(x))
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y)
    return torch.cat(all_preds), torch.cat(all_labels)


def evaluate_method(backbone, classifier, val_loaders, target_eval_loader,
                     method_name, device="cuda"):
    val_metrics = {d: evaluate_on_loader(backbone, classifier, l, device=device)
                    for d, l in val_loaders.items()}
    mean_source_acc = float(np.mean([val_metrics[d]["accuracy"] for d in val_metrics]))
    mean_source_f1 = float(np.mean([val_metrics[d]["macro_f1"] for d in val_metrics]))

    preds, labels = target_predictions(backbone, classifier, target_eval_loader, device=device)
    target_acc = float(accuracy_score(labels.numpy(), preds.numpy()))
    target_f1 = float(f1_score(labels.numpy(), preds.numpy(), average="macro", zero_division=0))
    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(len(PACS_CLASSES))))

    from task2.evaluation.class_analysis import per_class_accuracy
    per_class_acc = per_class_accuracy(preds, labels)

    return {
        "method": method_name,
        "source_val_by_domain": val_metrics,
        "mean_source_accuracy": mean_source_acc,
        "mean_source_macro_f1": mean_source_f1,
        "target_accuracy": target_acc,
        "target_macro_f1": target_f1,
        "per_class_target_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": PACS_CLASSES,
    }, preds, labels