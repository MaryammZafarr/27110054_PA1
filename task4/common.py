
import os

import numpy as np

from scores import LOGIT_SCORES, MahalanobisScorer, placeholder_score
from utils import CIFAR10_CLASSES, NUM_KNOWN

K = NUM_KNOWN
SPLITS = ["val", "test", "near", "far", "valmix"]


def load_bundle(cache_dir, name):
    kn = np.load(os.path.join(cache_dir, f"{name}_known.npz"))
    un = np.load(os.path.join(cache_dir, f"{name}_unknown.npz"))
    meta = np.load(os.path.join(cache_dir, "unknown_meta.npz"))
    near = meta["group"] == "near"
    b = dict(
        train=dict(logits=kn["train_logits"], feats=kn["train_feats"], labels=kn["train_labels"]),
        val=dict(logits=kn["val_logits"], feats=kn["val_feats"], labels=kn["val_labels"]),
        test=dict(logits=kn["test_logits"], feats=kn["test_feats"], labels=kn["test_labels"]),
        valmix=dict(logits=kn["valmix_logits"], feats=kn["valmix_feats"]),
        near=dict(logits=un["logits"][near], feats=un["feats"][near]),
        far=dict(logits=un["logits"][~near], feats=un["feats"][~near]),
        unk_all=dict(logits=un["logits"], feats=un["feats"]),
        meta={k: meta[k] for k in meta.files}, name=name, num_outputs=kn["test_logits"].shape[1])
    return b


def compute_scores(b):
    out = {}
    for nm, fn in LOGIT_SCORES.items():
        out[nm] = {s: fn(b[s]["logits"][:, :K]) for s in SPLITS + ["unk_all"]}
    mah = MahalanobisScorer().fit(b["train"]["feats"], b["train"]["labels"], K)
    out["Mahalanobis"] = {s: mah.score(b[s]["feats"]) for s in SPLITS + ["unk_all"]}
    if b["num_outputs"] > K:
        out["Placeholder"] = {s: placeholder_score(b[s]["logits"], K) for s in SPLITS + ["unk_all"]}
    return out


def csa(b):
    return float((b["test"]["logits"][:, :K].argmax(1) == b["test"]["labels"]).mean())


def pred_known(b, split):
    return b[split]["logits"][:, :K].argmax(1)


def class_names():
    return CIFAR10_CLASSES
