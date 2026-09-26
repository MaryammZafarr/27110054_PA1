

import json
import os
import random

import numpy as np
import torch
from torchvision.datasets import STL10

SEED = 6304
N_EVAL = 500
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer",
                  "dog", "horse", "monkey", "ship", "truck"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_eval_subset(root: str = "./data/stl10",
                       out_path: str = "./results/eval_subset_ids.json",
                       n_eval: int = N_EVAL):
    """Class-balanced subset of `n_eval` images from the official test split."""
    set_seed(SEED)
    test_set = STL10(root=root, split="test", download=True)
    labels = np.array(test_set.labels)
    n_classes = len(STL10_CLASSES)
    per_class = n_eval // n_classes
    remainder = n_eval - per_class * n_classes

    rng = np.random.RandomState(SEED)
    selected, class_counts = [], {}
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0].copy()
        rng.shuffle(idx_c)
        take = per_class + (1 if c < remainder else 0)
        if len(idx_c) < take:
            print(f"[WARN] class '{STL10_CLASSES[c]}' has only {len(idx_c)} "
                  f"test images (< {take} requested); using all available "
                  f"and documenting the imbalance.")
            take = len(idx_c)
        chosen = idx_c[:take]
        selected.extend(chosen.tolist())
        class_counts[STL10_CLASSES[c]] = int(take)

    selected = sorted(selected)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "seed": SEED,
            "n_requested": n_eval,
            "n_selected": len(selected),
            "class_counts": class_counts,
            "indices": selected,
        }, f, indent=2)
    print(f"Saved {len(selected)} eval image indices -> {out_path}")
    return selected


def build_train_val_split(root: str = "./data/stl10", val_frac: float = 0.2,
                           out_path: str = "./results/train_val_split.json"):
    """Stratified 80/20 split of the official train partition, seed 6304."""
    set_seed(SEED)
    train_set = STL10(root=root, split="train", download=True)
    labels = np.array(train_set.labels)
    n_classes = len(STL10_CLASSES)

    rng = np.random.RandomState(SEED)
    train_idx, val_idx = [], []
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0].copy()
        rng.shuffle(idx_c)
        n_val = int(round(len(idx_c) * val_frac))
        val_idx.extend(idx_c[:n_val].tolist())
        train_idx.extend(idx_c[n_val:].tolist())

    train_idx, val_idx = sorted(train_idx), sorted(val_idx)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "seed": SEED,
            "val_fraction": val_frac,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "train_indices": train_idx,
            "val_indices": val_idx,
        }, f, indent=2)
    print(f"Train/val split: {len(train_idx)} train / {len(val_idx)} val -> {out_path}")
    return train_idx, val_idx


if __name__ == "__main__":
    build_eval_subset()
    build_train_val_split()