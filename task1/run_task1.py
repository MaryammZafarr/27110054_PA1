
import json
import os
import sys

import numpy as np
import torch
from torchvision.datasets import STL10

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.transforms import to_common_tensor, translate_image
from data.make_subset import STL10_CLASSES, SEED, set_seed, build_eval_subset, build_train_val_split
from models.backbones import build_backbones
from analysis.evaluate_bias import (
    extract_features, train_linear_head, evaluate_clean, evaluate_color_bias,
    evaluate_translation, evaluate_patch_shuffle, evaluate_shape_texture,
)
from analysis.feature_similarity import cosine_stability
from analysis.representation import project_2d, plot_projection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "./data/stl10"
RESULTS_DIR = "./results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
CUE_DIR = os.path.join(RESULTS_DIR, "cue_conflicts")
CUE_META = os.path.join(CUE_DIR, "metadata.csv")


def load_images_by_indices(split: str, indices: list):
    ds = STL10(root=ROOT, split=split, download=True)
    imgs, labels = [], []
    for idx in indices:
        pil_img, lbl = ds[idx]
        imgs.append(to_common_tensor(pil_img))
        labels.append(lbl)
    return torch.stack(imgs, dim=0), torch.tensor(labels)


def load_cue_conflict_content_images(meta_csv: str, stl_root: str):
    import csv
    rows = list(csv.DictReader(open(meta_csv)))
    train_ds = STL10(root=stl_root, split="train", download=True)
    imgs = []
    for r in rows:
        pil_img, _ = train_ds[int(r["content_idx"])]
        imgs.append(to_common_tensor(pil_img))
    return torch.stack(imgs, dim=0)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    class_names = STL10_CLASSES

    subset_path = os.path.join(RESULTS_DIR, "eval_subset_ids.json")
    split_path = os.path.join(RESULTS_DIR, "train_val_split.json")
    if not os.path.exists(subset_path):
        build_eval_subset(root=ROOT, out_path=subset_path)
    if not os.path.exists(split_path):
        build_train_val_split(root=ROOT, out_path=split_path)

    eval_ids = json.load(open(subset_path))["indices"]
    split = json.load(open(split_path))
    train_ids, val_ids = split["train_indices"], split["val_indices"]

    print("Loading images")
    eval_images, eval_labels = load_images_by_indices("test", eval_ids)
    train_images, train_labels = load_images_by_indices("train", train_ids)
    val_images, val_labels = load_images_by_indices("train", val_ids)

    print("Building backbones")
    backbones = build_backbones(device=DEVICE)
    clip_backbone = backbones["clip_vitb32"]

    heads, head_val_acc = {}, {}
    for name, bb in backbones.items():
        print(f"[{name}] extracting train/val features")
        train_feats = extract_features(bb, train_images, device=DEVICE)
        val_feats = extract_features(bb, val_images, device=DEVICE)
        print(f"[{name}] training linear head")
        head, best_acc = train_linear_head(
            train_feats, train_labels, val_feats, val_labels,
            n_classes=len(class_names), device=DEVICE)
        heads[name] = head
        head_val_acc[name] = best_acc
        torch.save(head.state_dict(), os.path.join(RESULTS_DIR, f"head_{name}.pt"))
    save_json(head_val_acc, os.path.join(RESULTS_DIR, "head_val_accuracy.json"))

    print("Evaluating clean baseline")
    clean_results, clean_preds = evaluate_clean(
        heads, backbones, eval_images, eval_labels, clip_backbone, class_names, device=DEVICE)
    save_json(clean_results, os.path.join(RESULTS_DIR, "clean_baseline.json"))

    print("Evaluating color bias (grayscale + hue rotation)")
    color_results, color_variants = evaluate_color_bias(
        heads, backbones, eval_images, eval_labels, clip_backbone, class_names,
        clean_preds, hue_degrees=90.0, device=DEVICE)
    save_json(color_results, os.path.join(RESULTS_DIR, "color_bias.json"))

    print("Evaluating translation robustness (0/8/16/32 px, 4 directions)")
    translation_results = evaluate_translation(
        heads, backbones, eval_images, eval_labels, clip_backbone, class_names,
        displacements=(0, 8, 16, 32), device=DEVICE)
    save_json(translation_results, os.path.join(RESULTS_DIR, "translation.json"))

    print("Evaluating patch-shuffle robustness")
    patch_results, shuffled_images = evaluate_patch_shuffle(
        heads, backbones, eval_images, eval_labels, clip_backbone, class_names,
        clean_preds, grid=4, seed=SEED, device=DEVICE)
    save_json(patch_results, os.path.join(RESULTS_DIR, "patch_shuffle.json"))

    shape_texture_results, cue_imgs = None, None
    if os.path.exists(CUE_META):
        print("Evaluating shape-vs-texture cue conflicts")
        (shape_texture_results, st_preds, st_files,
         shape_lbls, texture_lbls, cue_imgs) = evaluate_shape_texture(
            heads, backbones, clip_backbone, class_names, CUE_DIR, CUE_META, device=DEVICE)
        save_json(shape_texture_results, os.path.join(RESULTS_DIR, "shape_texture.json"))

        per_image = []
        for i, fname in enumerate(st_files):
            row = {"filename": fname,
                   "shape_class": class_names[shape_lbls[i].item()],
                   "texture_class": class_names[texture_lbls[i].item()]}
            for name in list(backbones.keys()) + ["clip_zeroshot"]:
                row[f"pred_{name}"] = class_names[st_preds[name][i].item()]
            per_image.append(row)
        save_json(per_image, os.path.join(RESULTS_DIR, "shape_texture_per_image.json"))
    else:
        print(f"[SKIP] {CUE_META} not found")

    print("Computing representation stability (cosine similarity)")
    stability_results = {}
    translated_16 = translate_image(eval_images, 16, 0)
    paired_variants = {
        "grayscale": color_variants["grayscale"],
        "translation_16px": translated_16,
        "patch_shuffle": shuffled_images,
    }

    cue_content_images = None
    if shape_texture_results is not None:
        cue_content_images = load_cue_conflict_content_images(CUE_META, ROOT)

    for name, bb in backbones.items():
        stability_results[name] = {}
        clean_feats = extract_features(bb, eval_images, device=DEVICE)
        for variant_name, imgs in paired_variants.items():
            trans_feats = extract_features(bb, imgs, device=DEVICE)
            stability_results[name][variant_name] = cosine_stability(clean_feats, trans_feats)

        if cue_content_images is not None:
            content_feats = extract_features(bb, cue_content_images, device=DEVICE)
            cue_feats = extract_features(bb, cue_imgs, device=DEVICE)
            stability_results[name]["cue_conflict"] = cosine_stability(content_feats, cue_feats)

    save_json(stability_results, os.path.join(RESULTS_DIR, "representation_stability.json"))

    print("Fitting 2D projections")
    tsne_jobs = [
        ("translation16", translated_16, eval_images, eval_labels),
        ("grayscale", color_variants["grayscale"], eval_images, eval_labels),
        ("patch_shuffle", shuffled_images, eval_images, eval_labels),
    ]
    if cue_content_images is not None:
        tsne_jobs.append(("cue_conflict", cue_imgs, cue_content_images, shape_lbls))

    for name, bb in backbones.items():
        for vname, t_imgs, c_imgs, lbls in tsne_jobs:
            c_feats = extract_features(bb, c_imgs, device=DEVICE).numpy()
            t_feats = extract_features(bb, t_imgs, device=DEVICE).numpy()
            combined = np.concatenate([c_feats, t_feats], axis=0)
            coords = project_2d(combined, method="tsne", seed=SEED)
            n = c_feats.shape[0]
            plot_projection(
                coords[:n], coords[n:], lbls.numpy(), lbls.numpy(),
                class_names, title=f"{name}: clean vs {vname} (tsne)",
                save_path=os.path.join(FIG_DIR, f"tsne_{name}_{vname}.png"))


if __name__ == "__main__":
    main()
