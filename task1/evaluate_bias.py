
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from data.transforms import (to_common_tensor, to_grayscale, hue_rotate,
                              translate_all_directions, patch_shuffle)
from models.backbones import LinearHead

SEED = 6304


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Feature extraction (call once per backbone / image set, then reuse)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(backbone, images: torch.Tensor, batch_size: int = 64,
                      device: str = "cuda") -> torch.Tensor:
    backbone.eval()
    feats = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device)
        feats.append(backbone(batch).cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def clip_zero_shot_logits(clip_backbone, images: torch.Tensor, class_names,
                           batch_size: int = 64, device: str = "cuda") -> torch.Tensor:
    logits = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device)
        logits.append(clip_backbone.zero_shot_logits(batch, class_names, device=device).cpu())
    return torch.cat(logits, dim=0)


# --------------------------------------------------------------------------- #
# Linear head training with early stopping
# --------------------------------------------------------------------------- #
def train_linear_head(train_feats, train_labels, val_feats, val_labels, n_classes: int,
                       device: str = "cuda", lr: float = 1e-3, weight_decay: float = 1e-4,
                       max_epochs: int = 50, patience: int = 5, batch_size: int = 128,
                       seed: int = SEED):
    set_seed(seed)
    head = LinearHead(train_feats.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_feats, train_labels = train_feats.to(device), train_labels.to(device)
    val_feats, val_labels = val_feats.to(device), val_labels.to(device)

    best_val_acc, epochs_no_improve = -1.0, 0
    best_state = copy.deepcopy(head.state_dict())
    n = train_feats.shape[0]

    for epoch in range(max_epochs):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(head(train_feats[idx]), train_labels[idx])
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_pred = head(val_feats).argmax(dim=1)
            val_acc = (val_pred == val_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc, epochs_no_improve = val_acc, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    early stop at epoch {epoch + 1} (best val acc {best_val_acc:.4f})")
                break

    head.load_state_dict(best_state)
    return head, best_val_acc


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    macro_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
    mean_conf = probs.max(dim=1).values.mean().item()
    return {"top1_acc": acc, "macro_f1": macro_f1, "mean_confidence": mean_conf}, preds


def prediction_consistency(preds_a: torch.Tensor, preds_b: torch.Tensor) -> float:
    return (preds_a == preds_b).float().mean().item()


# --------------------------------------------------------------------------- #
# 1. Clean baseline
# --------------------------------------------------------------------------- #
def evaluate_clean(heads, backbones, eval_images, eval_labels, clip_backbone,
                    class_names, device: str = "cuda"):
    results, preds_by_model = {}, {}
    for name, backbone in backbones.items():
        feats = extract_features(backbone, eval_images, device=device)
        logits = heads[name](feats.to(device)).cpu()
        m, preds = compute_metrics(logits, eval_labels)
        results[name], preds_by_model[name] = m, preds

    zs_logits = clip_zero_shot_logits(clip_backbone, eval_images, class_names, device=device)
    m, preds = compute_metrics(zs_logits, eval_labels)
    results["clip_zeroshot"], preds_by_model["clip_zeroshot"] = m, preds
    return results, preds_by_model


# --------------------------------------------------------------------------- #
# 2. Color bias (grayscale + hue rotation)
# --------------------------------------------------------------------------- #
def evaluate_color_bias(heads, backbones, eval_images, eval_labels, clip_backbone,
                         class_names, clean_preds, hue_degrees: float = 90.0,
                         device: str = "cuda"):
    variants = {
        "grayscale": to_grayscale(eval_images),
        "hue_rotate": hue_rotate(eval_images, degrees=hue_degrees),
    }
    results = {}
    for variant_name, imgs in variants.items():
        results[variant_name] = {}
        for name, backbone in backbones.items():
            feats = extract_features(backbone, imgs, device=device)
            logits = heads[name](feats.to(device)).cpu()
            m, preds = compute_metrics(logits, eval_labels)
            m["consistency_vs_clean"] = prediction_consistency(preds, clean_preds[name])
            results[variant_name][name] = m

        zs_logits = clip_zero_shot_logits(clip_backbone, imgs, class_names, device=device)
        m, preds = compute_metrics(zs_logits, eval_labels)
        m["consistency_vs_clean"] = prediction_consistency(preds, clean_preds["clip_zeroshot"])
        results[variant_name]["clip_zeroshot"] = m
    return results, variants


# --------------------------------------------------------------------------- #
# 3. Translation
# --------------------------------------------------------------------------- #
def evaluate_translation(heads, backbones, eval_images, eval_labels, clip_backbone,
                          class_names, displacements=(0, 8, 16, 32), device: str = "cuda"):
    """
    For each displacement, average accuracy/consistency across the 4
    cardinal directions. Returns {model_name: {delta: {"accuracy", "consistency"}}}.
    """
    model_names = list(backbones.keys()) + ["clip_zeroshot"]
    results = {name: {} for name in model_names}

    clean_preds = {}
    for name, backbone in backbones.items():
        feats = extract_features(backbone, eval_images, device=device)
        logits = heads[name](feats.to(device)).cpu()
        _, clean_preds[name] = compute_metrics(logits, eval_labels)
    zs_logits = clip_zero_shot_logits(clip_backbone, eval_images, class_names, device=device)
    _, clean_preds["clip_zeroshot"] = compute_metrics(zs_logits, eval_labels)

    for delta in displacements:
        acc_by_model = {name: [] for name in model_names}
        cons_by_model = {name: [] for name in model_names}
        for _, shifted in translate_all_directions(eval_images, delta):
            for name, backbone in backbones.items():
                feats = extract_features(backbone, shifted, device=device)
                logits = heads[name](feats.to(device)).cpu()
                m, preds = compute_metrics(logits, eval_labels)
                acc_by_model[name].append(m["top1_acc"])
                cons_by_model[name].append(prediction_consistency(preds, clean_preds[name]))

            zs_logits = clip_zero_shot_logits(clip_backbone, shifted, class_names, device=device)
            m, preds = compute_metrics(zs_logits, eval_labels)
            acc_by_model["clip_zeroshot"].append(m["top1_acc"])
            cons_by_model["clip_zeroshot"].append(
                prediction_consistency(preds, clean_preds["clip_zeroshot"]))

        for name in model_names:
            results[name][delta] = {
                "accuracy": float(np.mean(acc_by_model[name])),
                "consistency": float(np.mean(cons_by_model[name])),
            }
    return results


# --------------------------------------------------------------------------- #
# 4. Patch structure (4x4 shuffle)
# --------------------------------------------------------------------------- #
def evaluate_patch_shuffle(heads, backbones, eval_images, eval_labels, clip_backbone,
                            class_names, clean_preds, grid: int = 4, seed: int = SEED,
                            device: str = "cuda"):
    shuffled = patch_shuffle(eval_images, grid=grid, seed=seed)
    results = {}
    for name, backbone in backbones.items():
        feats = extract_features(backbone, shuffled, device=device)
        logits = heads[name](feats.to(device)).cpu()
        m, preds = compute_metrics(logits, eval_labels)
        m["consistency_vs_clean"] = prediction_consistency(preds, clean_preds[name])
        results[name] = m

    zs_logits = clip_zero_shot_logits(clip_backbone, shuffled, class_names, device=device)
    m, preds = compute_metrics(zs_logits, eval_labels)
    m["consistency_vs_clean"] = prediction_consistency(preds, clean_preds["clip_zeroshot"])
    results["clip_zeroshot"] = m
    return results, shuffled


# --------------------------------------------------------------------------- #
# 5. Shape vs. texture (cue-conflict) evaluation
# --------------------------------------------------------------------------- #
def _shape_texture_counts(preds: torch.Tensor, shape_labels: torch.Tensor,
                           texture_labels: torch.Tensor) -> dict:
    n_total = preds.shape[0]
    n_shape = (preds == shape_labels).sum().item()
    n_texture = (preds == texture_labels).sum().item()
    n_other = n_total - n_shape - n_texture
    denom = n_shape + n_texture
    shape_bias = 100.0 * n_shape / denom if denom > 0 else float("nan")
    coverage = 100.0 * denom / n_total if n_total > 0 else float("nan")
    return {"n_total": n_total, "n_shape": n_shape, "n_texture": n_texture,
            "n_other": n_other, "shape_bias_pct": shape_bias, "coverage_pct": coverage}


def evaluate_shape_texture(heads, backbones, clip_backbone, class_names,
                            cue_conflict_dir: str, meta_csv: str, device: str = "cuda"):
    """
    Loads cue-conflict images + metadata and classifies each model's
    prediction as shape / texture / other.
    Returns: (results_by_model, preds_by_model, filenames, shape_labels,
              texture_labels, images)
    """
    import csv
    from PIL import Image

    rows = list(csv.DictReader(open(meta_csv)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    imgs, shape_labels, texture_labels, filenames = [], [], [], []
    for r in rows:
        img = Image.open(os.path.join(cue_conflict_dir, r["filename"])).convert("RGB")
        imgs.append(to_common_tensor(img))
        shape_labels.append(class_to_idx[r["shape_class"]])
        texture_labels.append(class_to_idx[r["texture_class"]])
        filenames.append(r["filename"])
    imgs = torch.stack(imgs, dim=0)
    shape_labels = torch.tensor(shape_labels)
    texture_labels = torch.tensor(texture_labels)

    results, preds_by_model = {}, {}
    for name, backbone in backbones.items():
        feats = extract_features(backbone, imgs, device=device)
        logits = heads[name](feats.to(device)).cpu()
        preds = logits.argmax(dim=1)
        preds_by_model[name] = preds
        results[name] = _shape_texture_counts(preds, shape_labels, texture_labels)

    zs_logits = clip_zero_shot_logits(clip_backbone, imgs, class_names, device=device)
    preds = zs_logits.argmax(dim=1)
    preds_by_model["clip_zeroshot"] = preds
    results["clip_zeroshot"] = _shape_texture_counts(preds, shape_labels, texture_labels)

    return results, preds_by_model, filenames, shape_labels, texture_labels, imgs