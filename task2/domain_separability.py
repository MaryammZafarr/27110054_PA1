
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from shared.pacs import SEED


@torch.no_grad()
def extract_backbone_features(backbone, loader, device="cuda"):
    backbone.eval()
    feats = []
    for x, _ in loader:
        feats.append(backbone(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def domain_separability_score(backbone, val_loaders: dict, target_eval_loader,
                               device="cuda", seed: int = SEED) -> dict:
    source_feats = np.concatenate(
        [extract_backbone_features(backbone, l, device=device) for l in val_loaders.values()],
        axis=0)
    target_feats_all = extract_backbone_features(backbone, target_eval_loader, device=device)

    rng = np.random.RandomState(seed)
    n = min(len(source_feats), len(target_feats_all))
    src_idx = rng.choice(len(source_feats), size=n, replace=False)
    tgt_idx = rng.choice(len(target_feats_all), size=n, replace=False)
    source_feats, target_feats = source_feats[src_idx], target_feats_all[tgt_idx]

    X = np.concatenate([source_feats, target_feats], axis=0)
    y = np.concatenate([np.zeros(len(source_feats)), np.ones(len(target_feats))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    held_out_acc = float(clf.score(X_test, y_test))

    return {
        "domain_separability_accuracy": held_out_acc,
        "n_source_features_used": int(n),
        "n_target_features_used": int(n),
        "chance_level": 0.5,
    }