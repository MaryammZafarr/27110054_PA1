
import copy
import os

import numpy as np
import torch
import torch.nn as nn

from shared.pacs_protocol import (
    InfiniteLoader, set_seed, set_train_bn_frozen, evaluate_on_loader, save_json,
)

CE = nn.CrossEntropyLoss()


class SAM:

    def __init__(self, params, base_optimizer_cls, rho=0.05, **base_kwargs):
        self.params = [p for p in params if p.requires_grad]
        self.rho = rho
        self.base_optimizer = base_optimizer_cls(self.params, **base_kwargs)
        self._e_ws = []

    @torch.no_grad()
    def ascent_step(self):
        norms = [p.grad.norm(2) for p in self.params if p.grad is not None]
        grad_norm = torch.norm(torch.stack(norms), 2) if norms else torch.tensor(0.0)
        scale = self.rho / (grad_norm + 1e-12)
        self._e_ws = []
        for p in self.params:
            if p.grad is None:
                self._e_ws.append(None)
                continue
            e_w = p.grad * scale
            p.add_(e_w)
            self._e_ws.append(e_w)

    @torch.no_grad()
    def descent_step(self):
        for p, e_w in zip(self.params, self._e_ws):
            if e_w is not None:
                p.sub_(e_w)
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad(set_to_none=True)


def train(backbone, classifier, source_loaders, val_loaders, steps_per_epoch,
          results_dir, rho=0.05, max_epochs=30, lr=1e-4, weight_decay=1e-4,
          patience=5, device="cuda", seed=6304, method_name="sam", **_ignored):
    set_seed(seed)
    params = list(backbone.parameters()) + list(classifier.parameters())
    optimizer = SAM(params, torch.optim.AdamW, rho=rho, lr=lr, weight_decay=weight_decay)

    src_iters = {d: InfiniteLoader(l) for d, l in source_loaders.items()}

    best_mean_val_f1 = -1.0
    epochs_no_improve = 0
    best_state = {
        "backbone": copy.deepcopy(backbone.state_dict()),
        "classifier": copy.deepcopy(classifier.state_dict()),
    }
    history = []

    def compute_cls_loss():
        source_batches = {d: next(it) for d, it in src_iters.items()}
        xs, ys = [], []
        for _, (x, y) in source_batches.items():
            xs.append(x)
            ys.append(y)
        x = torch.cat(xs, dim=0).to(device, non_blocking=True)
        y = torch.cat(ys, dim=0).to(device, non_blocking=True)
        logits = classifier(backbone(x))
        return CE(logits, y)

    for epoch in range(max_epochs):
        set_train_bn_frozen(backbone, classifier)
        loss_sum, loss_count = 0.0, 0

        for _ in range(steps_per_epoch):
            # pass 1: gradient at theta -> ascend to theta + e_w
            optimizer.zero_grad()
            loss1 = compute_cls_loss()
            loss1.backward()
            optimizer.ascent_step()

            # pass 2: gradient AT the perturbed point, restore theta, apply
            set_train_bn_frozen(backbone, classifier)  # keep BN stats frozen at perturbed point too
            optimizer.zero_grad()
            loss2 = compute_cls_loss()
            loss2.backward()
            optimizer.descent_step()

            loss_sum += loss2.item()
            loss_count += 1

        epoch_avg_loss = loss_sum / loss_count

        val_metrics = {d: evaluate_on_loader(backbone, classifier, l, device=device)
                       for d, l in val_loaders.items()}
        mean_val_f1 = float(np.mean([val_metrics[d]["macro_f1"] for d in val_metrics]))

        history.append({"epoch": epoch + 1, "train_losses": {"cls_loss": epoch_avg_loss},
                         "val_metrics": val_metrics, "mean_val_macro_f1": mean_val_f1})
        print(f"[{method_name}] epoch {epoch+1}/{max_epochs} "
              f"mean_val_macro_f1={mean_val_f1:.4f} cls_loss={epoch_avg_loss:.4f}")

        if mean_val_f1 > best_mean_val_f1:
            best_mean_val_f1 = mean_val_f1
            epochs_no_improve = 0
            best_state = {
                "backbone": copy.deepcopy(backbone.state_dict()),
                "classifier": copy.deepcopy(classifier.state_dict()),
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{method_name}] early stop at epoch {epoch+1} "
                      f"(best mean val macro-F1 {best_mean_val_f1:.4f})")
                break

    backbone.load_state_dict(best_state["backbone"])
    classifier.load_state_dict(best_state["classifier"])

    os.makedirs(results_dir, exist_ok=True)
    save_json(history, os.path.join(results_dir, f"{method_name}_training_curve.json"))
    torch.save(best_state["backbone"], os.path.join(results_dir, f"{method_name}_backbone.pt"))
    torch.save(best_state["classifier"], os.path.join(results_dir, f"{method_name}_classifier.pt"))

    return {"best_mean_val_macro_f1": best_mean_val_f1, "history": history}
