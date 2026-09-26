
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

from shared.pacs import SEED, PACS_CLASSES, SOURCE_DOMAINS, TARGET_DOMAIN, \
    list_domain_files, PACSDomainDataset, train_transform, eval_transform

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_batchnorm_running_stats(model: nn.Module):
    """After model.train(), put every BatchNorm module into eval mode so
    running_mean/running_var stay frozen at their pretrained ImageNet
    values, while gamma/beta stay trainable and everything else stays in
    train mode. Call every time right after .train()."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def set_train_bn_frozen(*modules: nn.Module):
    for m in modules:
        m.train()
        freeze_batchnorm_running_stats(m)


class InfiniteLoader:

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iterator = iter(self.dataloader)

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)


def build_source_splits(pacs_root: str, splits_path: str, val_frac: float = 0.2,
                         seed: int = SEED):
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            return json.load(f)["splits"]

    random.seed(seed)
    np.random.seed(seed)
    splits = {}
    for domain in SOURCE_DOMAINS:
        files, labels = list_domain_files(pacs_root, domain)
        labels_arr = np.array(labels)
        rng = np.random.RandomState(seed)
        train_idx, val_idx = [], []
        for c in range(len(PACS_CLASSES)):
            idx_c = np.where(labels_arr == c)[0].copy()
            rng.shuffle(idx_c)
            n_val = int(round(len(idx_c) * val_frac))
            val_idx.extend(idx_c[:n_val].tolist())
            train_idx.extend(idx_c[n_val:].tolist())
        splits[domain] = {
            "n_total": len(files),
            "train_indices": sorted(train_idx),
            "val_indices": sorted(val_idx),
        }
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump({"seed": seed, "val_fraction": val_frac, "splits": splits}, f, indent=2)
    print(f"[splits] wrote fresh source train/val splits -> {splits_path}")
    return splits


def build_datasets(pacs_root: str, splits: dict):
   
    train_datasets, val_datasets = {}, {}
    for domain in SOURCE_DOMAINS:
        files, labels = list_domain_files(pacs_root, domain)
        train_datasets[domain] = PACSDomainDataset(
            files, labels, splits[domain]["train_indices"], train_transform)
        val_datasets[domain] = PACSDomainDataset(
            files, labels, splits[domain]["val_indices"], eval_transform)

    tgt_files, tgt_labels = list_domain_files(pacs_root, TARGET_DOMAIN)
    all_idx = list(range(len(tgt_files)))
    target_dataset_train = PACSDomainDataset(tgt_files, tgt_labels, all_idx, train_transform)
    target_dataset_eval = PACSDomainDataset(tgt_files, tgt_labels, all_idx, eval_transform)

    return train_datasets, val_datasets, target_dataset_train, target_dataset_eval


def build_loaders(train_datasets, val_datasets, target_dataset_train, target_dataset_eval,
                   source_batch_per_domain: int = 8, target_batch_size: int = 24,
                   num_workers: int = 4):

    persistent = num_workers > 0
    source_train_loaders = {
        d: DataLoader(ds, batch_size=source_batch_per_domain, shuffle=True,
                       num_workers=num_workers, drop_last=True, pin_memory=True,
                       persistent_workers=persistent)
        for d, ds in train_datasets.items()
    }
    target_loader = DataLoader(target_dataset_train, batch_size=target_batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=True, pin_memory=True,
                                persistent_workers=persistent)
    target_eval_loader = DataLoader(target_dataset_eval, batch_size=64, shuffle=False,
                                     num_workers=num_workers)
    val_loaders = {
        d: DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)
        for d, ds in val_datasets.items()
    }
    return source_train_loaders, target_loader, target_eval_loader, val_loaders



def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    preds_np, labels_np = preds.cpu().numpy(), labels.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "macro_f1": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate_on_loader(backbone, classifier, loader, device: str = "cuda") -> dict:
    backbone.eval()
    classifier.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = classifier(backbone(x))
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y)
    return compute_metrics(torch.cat(all_preds), torch.cat(all_labels))


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)



def run_training(backbone, classifier, aux_module, compute_losses,
                  source_loaders: dict, target_loader, val_loaders: dict,
                  method_name: str, results_dir: str, steps_per_epoch: int,
                  max_epochs: int = 30, lr: float = 1e-4, weight_decay: float = 1e-4,
                  patience: int = 5, device: str = "cuda", seed: int = SEED,
                  use_amp: bool = False):
   
    set_seed(seed)
    params = list(backbone.parameters()) + list(classifier.parameters())
    if aux_module is not None:
        params += list(aux_module.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    src_iters = {d: InfiniteLoader(l) for d, l in source_loaders.items()}
    tgt_iter = InfiniteLoader(target_loader) if target_loader is not None else None

    best_mean_val_f1 = -1.0
    epochs_no_improve = 0
    best_state = {
        "backbone": copy.deepcopy(backbone.state_dict()),
        "classifier": copy.deepcopy(classifier.state_dict()),
        "aux": copy.deepcopy(aux_module.state_dict()) if aux_module is not None else None,
    }
    history = []
    total_steps = max_epochs * steps_per_epoch
    global_step = 0

    for epoch in range(max_epochs):
        set_train_bn_frozen(backbone, classifier, *([aux_module] if aux_module is not None else []))
        loss_sums, loss_counts = {}, {}

        for _ in range(steps_per_epoch):
            p = global_step / max(1, total_steps - 1)
            source_batches = {d: next(it) for d, it in src_iters.items()}
            target_batch = next(tgt_iter) if tgt_iter is not None else None

            optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(backbone, classifier, aux_module,
                                     source_batches, target_batch, p, device)
            
            losses["total"].backward()
            all_params = []
            for pg in optimizer.param_groups:
                all_params.extend(pg["params"])
            total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            if not torch.isfinite(total_norm):
                print(f"[{method_name}] step {global_step}: non-finite grad norm, p={p:.3f}, epoch={epoch+1}")
            optimizer.step()
            
            
            for k, v in losses.items():
                v_float = v.item() if torch.is_tensor(v) else float(v)
                loss_sums[k] = loss_sums.get(k, 0.0) + v_float
                loss_counts[k] = loss_counts.get(k, 0) + 1
            global_step += 1

        epoch_avg_losses = {k: loss_sums[k] / loss_counts[k] for k in loss_sums}

        val_metrics = {d: evaluate_on_loader(backbone, classifier, l, device=device)
                        for d, l in val_loaders.items()}
        mean_val_f1 = float(np.mean([val_metrics[d]["macro_f1"] for d in val_metrics]))

        history.append({"epoch": epoch + 1, "train_losses": epoch_avg_losses,
                         "val_metrics": val_metrics, "mean_val_macro_f1": mean_val_f1})
        print(f"[{method_name}] epoch {epoch+1}/{max_epochs} "
              f"mean_val_macro_f1={mean_val_f1:.4f} "
              f"losses={ {k: round(v,4) for k,v in epoch_avg_losses.items()} }")

        if mean_val_f1 > best_mean_val_f1:
            best_mean_val_f1 = mean_val_f1
            epochs_no_improve = 0
            best_state = {
                "backbone": copy.deepcopy(backbone.state_dict()),
                "classifier": copy.deepcopy(classifier.state_dict()),
                "aux": copy.deepcopy(aux_module.state_dict()) if aux_module is not None else None,
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{method_name}] early stop at epoch {epoch+1} "
                      f"(best mean val macro-F1 {best_mean_val_f1:.4f})")
                break

    backbone.load_state_dict(best_state["backbone"])
    classifier.load_state_dict(best_state["classifier"])
    if aux_module is not None and best_state["aux"] is not None:
        aux_module.load_state_dict(best_state["aux"])

    os.makedirs(results_dir, exist_ok=True)
    save_json(history, os.path.join(results_dir, f"{method_name}_training_curve.json"))
    torch.save(best_state["backbone"], os.path.join(results_dir, f"{method_name}_backbone.pt"))
    torch.save(best_state["classifier"], os.path.join(results_dir, f"{method_name}_classifier.pt"))
    if aux_module is not None:
        torch.save(best_state["aux"], os.path.join(results_dir, f"{method_name}_aux.pt"))

    return {"best_mean_val_macro_f1": best_mean_val_f1, "history": history}