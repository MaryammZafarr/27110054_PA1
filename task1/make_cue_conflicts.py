
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image

SEED = 6304
STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1"]
CONTENT_LAYER = "relu4_1"

STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer",
                  "dog", "horse", "monkey", "ship", "truck"]

# 5 unordered class pairs; both shape/texture directions are generated for each.
CLASS_PAIRS = [
    ("cat", "dog"),
    ("horse", "deer"),
    ("car", "truck"),
    ("bird", "airplane"),
    ("monkey", "cat"),
]

_LAYER_NAME_MAP = {
    "0": "conv1_1", "1": "relu1_1", "2": "conv1_2", "3": "relu1_2", "4": "pool1",
    "5": "conv2_1", "6": "relu2_1", "7": "conv2_2", "8": "relu2_2", "9": "pool2",
    "10": "conv3_1", "11": "relu3_1", "12": "conv3_2", "13": "relu3_2",
    "14": "conv3_3", "15": "relu3_3", "16": "conv3_4", "17": "relu3_4", "18": "pool3",
    "19": "conv4_1", "20": "relu4_1", "21": "conv4_2", "22": "relu4_2",
    "23": "conv4_3", "24": "relu4_3", "25": "conv4_4", "26": "relu4_4", "27": "pool4",
}


class VGGFeatures(nn.Module):
    """Wraps a frozen, pretrained torchvision VGG19, exposing named layers."""

    def __init__(self, device="cuda"):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = vgg[:28]  # up through relu4_1 (and a little beyond)
        for p in self.layers.parameters():
            p.requires_grad = False
        self.layers.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats, out = {}, x
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            name = _LAYER_NAME_MAP.get(str(idx))
            if name is not None:
                feats[name] = out
        return feats


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Channel-wise AdaIN: renormalize content feature stats to style stats."""
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + eps
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True) + eps
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def stylize(content_img: torch.Tensor, style_img: torch.Tensor, vgg: VGGFeatures,
            alpha: float = 1.0, n_steps: int = 300, style_weight: float = 1e4,
            content_weight: float = 1.0, lr: float = 0.05, device: str = "cuda") -> torch.Tensor:
    """
    Produce one cue-conflict image: shape/content = content_img,
    texture/style = style_img. `alpha` blends the AdaIN target with the raw
    content feature (alpha=1.0 -> full style-statistics transfer).
    content_img, style_img: (1,3,224,224) in [0,1].
    """
    content_img = content_img.to(device)
    style_img = style_img.to(device)
    with torch.no_grad():
        c_feats = vgg(content_img)
        s_feats = vgg(style_img)
        target_feat = adain(c_feats[CONTENT_LAYER], s_feats[CONTENT_LAYER])
        target_feat = alpha * target_feat + (1 - alpha) * c_feats[CONTENT_LAYER]
        style_grams = {l: gram_matrix(s_feats[l]) for l in STYLE_LAYERS}

    out = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([out], lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        feats = vgg(out.clamp(0, 1))
        content_loss = F.mse_loss(feats[CONTENT_LAYER], target_feat)
        style_loss = sum(F.mse_loss(gram_matrix(feats[l]), style_grams[l]) for l in STYLE_LAYERS)
        loss = content_weight * content_loss + style_weight * style_loss
        loss.backward()
        optimizer.step()
    return out.detach().clamp(0, 1).cpu()


def visual_rejection_rule(stylized: torch.Tensor, content: torch.Tensor,
                           min_std: float = 0.03, max_identity_cosine: float = 0.995) -> bool:
    """
    Purely VISUAL acceptance rule (never uses model predictions, per spec):
      - reject near-uniform / collapsed outputs (optimization diverged or
        collapsed to a flat color),
      - reject outputs that are almost pixel-identical to the content image
        (style was effectively not applied),
      - reject any NaNs.
    Returns True if the image should be ACCEPTED.
    """
    if torch.isnan(stylized).any():
        return False
    if stylized.std().item() < min_std:
        return False
    cos_sim = F.cosine_similarity(
        stylized.flatten().unsqueeze(0), content.flatten().unsqueeze(0)
    ).item()
    if cos_sim > max_identity_cosine:
        return False
    return True


def generate_cue_conflicts(dataset, class_to_idx: dict, out_dir: str = "./results/cue_conflicts",
                            n_per_direction: int = 25, alpha: float = 1.0, device: str = "cuda"):
    """
    dataset: an indexable dataset returning (PIL image, int label); must also
             expose a `.labels` array/list (as torchvision's STL10 does) for
             fast class-based indexing.
    class_to_idx: {class_name: int_label}
    n_per_direction: attempts capped at 3x this value per (pair, direction);
                      5 pairs x 2 directions x 25 accepted = 250 >= the
                      required 200 valid conflicts, with headroom for rejects.
    """
    from data.transforms import to_common_tensor  # local import avoids circularity

    os.makedirs(out_dir, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    vgg = VGGFeatures(device=device)

    labels_arr = np.array(dataset.labels)
    by_class = {c: np.where(labels_arr == c)[0].tolist() for c in class_to_idx.values()}

    meta_rows = []
    existing_files = [f for f in os.listdir(out_dir) if f.endswith(".png") and f.startswith("cc_")]
    n_accepted = len(existing_files)
    n_rejected = 0
    img_counter = n_accepted

    for cls_a, cls_b in CLASS_PAIRS:
        if n_accepted >= 200:
            break
        for shape_cls, texture_cls in [(cls_a, cls_b), (cls_b, cls_a)]:
            if n_accepted >= 200:
                break
            shape_pool = by_class[class_to_idx[shape_cls]][:]
            texture_pool = by_class[class_to_idx[texture_cls]][:]
            random.shuffle(shape_pool)
            random.shuffle(texture_pool)

            n_ok, attempt = 0, 0
            max_attempts = n_per_direction * 3
            while n_ok < n_per_direction and attempt < max_attempts:
                s_i = shape_pool[attempt % len(shape_pool)]
                t_i = texture_pool[attempt % len(texture_pool)]
                attempt += 1

                content_pil, _ = dataset[s_i]
                style_pil, _ = dataset[t_i]
                content_t = to_common_tensor(content_pil).unsqueeze(0)
                style_t = to_common_tensor(style_pil).unsqueeze(0)

                stylized = stylize(content_t, style_t, vgg, alpha=alpha, device=device)
                if visual_rejection_rule(stylized[0], content_t[0]):
                    fname = f"cc_{img_counter:04d}_{shape_cls}-shape_{texture_cls}-texture.png"
                    fpath = os.path.join(out_dir, fname)
                    arr = (stylized[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    Image.fromarray(arr).save(fpath)
                    meta_rows.append({
                        "filename": fname,
                        "shape_class": shape_cls,
                        "texture_class": texture_cls,
                        "content_idx": s_i,
                        "style_idx": t_i,
                    })
                    n_ok += 1
                    n_accepted += 1
                    img_counter += 1
                else:
                    n_rejected += 1

            print(f"  {shape_cls}(shape) x {texture_cls}(texture): "
                  f"{n_ok}/{n_per_direction} accepted after {attempt} attempts")

    meta_path = os.path.join(out_dir, "metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "shape_class", "texture_class", "content_idx", "style_idx"])
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"\nCue conflicts: {n_accepted} accepted, {n_rejected} rejected "
          f"(requirement: >= 200 accepted).")
    if n_accepted < 200:
        print("[WARN] fewer than 200 accepted — raise n_per_direction or "
              "relax the rejection thresholds in configs/config.yaml.")
    return meta_path


if __name__ == "__main__":
    from torchvision.datasets import STL10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_to_idx = {c: i for i, c in enumerate(STL10_CLASSES)}
    train_ds = STL10(root="./data/stl10", split="train", download=True)
    generate_cue_conflicts(train_ds, class_to_idx, out_dir="./results/cue_conflicts",
                            n_per_direction=25, alpha=1.0, device=device)