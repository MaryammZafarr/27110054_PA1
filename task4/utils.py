"""Shared helpers: seeding, config loading, hashing, small IO utilities.

Task 4 (Open-Set Recognition) -- every random source is seeded with SEED=6304 as the manual requires.
"""
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
import yaml

SEED = 6304
NUM_KNOWN = 10
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def set_seed(seed: int = SEED):
    """Seed python / numpy / torch (CPU+CUDA).

    DECISION: we do NOT force cudnn.deterministic (it slows ResNet training ~20-30% and the manual only
    asks for a fixed seed, not bit-wise determinism). We enable cudnn.benchmark for speed. Consequence:
    re-running gives statistically equivalent but not bit-identical numbers -> report this as a limitation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_overrides(pairs):
    """['epochs=2', 'lr=0.01'] -> {'epochs': 2, 'lr': 0.01} (YAML-typed values)."""
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        cfg[k] = v
    return cfg


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0
