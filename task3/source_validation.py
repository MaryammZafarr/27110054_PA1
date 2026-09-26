
import numpy as np

from shared.pacs_protocol import evaluate_on_loader


def mean_source_macro_f1(backbone, classifier, source_val_loaders: dict, device: str = "cuda") -> float:
    per_domain = {d: evaluate_on_loader(backbone, classifier, l, device=device)
                  for d, l in source_val_loaders.items()}
    return float(np.mean([m["macro_f1"] for m in per_domain.values()]))
