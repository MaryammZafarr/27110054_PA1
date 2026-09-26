
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@torch.no_grad()
def compute_source_domain_separability(backbone, val_loaders: dict, seed: int = 6304,
                                        device: str = "cuda") -> dict:
    backbone.eval()
    domain_names = sorted(val_loaders.keys())
    n_per_domain = min(len(l.dataset) for l in val_loaders.values())

    feats, labels = [], []
    for domain_idx, domain in enumerate(domain_names):
        loader = val_loaders[domain]
        collected = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            f = backbone(x).cpu().numpy()
            take = min(len(f), n_per_domain - collected)
            feats.append(f[:take])
            labels.extend([domain_idx] * take)
            collected += take
            if collected >= n_per_domain:
                break

    X = np.concatenate(feats, axis=0)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X_train, y_train)
    held_out_acc = clf.score(X_test, y_test)

    return {"domains": domain_names, "n_per_domain": n_per_domain,
            "held_out_accuracy": float(held_out_acc), "chance_accuracy": 1.0 / len(domain_names)}
