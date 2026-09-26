
import torch
import torch.nn as nn

from shared.pacs_protocol import run_training

CE = nn.CrossEntropyLoss()


def compute_losses(backbone, classifier, aux_module, source_batches, target_batch, p, device):
    xs, ys = [], []
    for _, (x, y) in source_batches.items():
        xs.append(x)
        ys.append(y)
    x = torch.cat(xs, dim=0).to(device, non_blocking=True)
    y = torch.cat(ys, dim=0).to(device, non_blocking=True)

    logits = classifier(backbone(x))
    loss = CE(logits, y)
    return {"total": loss, "cls_loss": loss}


def train(backbone, classifier, source_loaders, val_loaders, steps_per_epoch,
          results_dir, device="cuda", **kwargs):
    return run_training(
        backbone, classifier, aux_module=None, compute_losses=compute_losses,
        source_loaders=source_loaders, target_loader=None, val_loaders=val_loaders,
        method_name="source_only", results_dir=results_dir,
        steps_per_epoch=steps_per_epoch, device=device, **kwargs)