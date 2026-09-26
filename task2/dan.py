
import torch
import torch.nn as nn

from shared.pacs_protocol import run_training
from shared.mmd import rbf_mmd2

CE = nn.CrossEntropyLoss()
LAMBDA_MMD = 1.0


def make_compute_losses(lambda_mmd: float = LAMBDA_MMD):
    def compute_losses(backbone, classifier, aux_module, source_batches, target_batch, p, device):
        xs, ys = [], []
        for _, (x, y) in source_batches.items():
            xs.append(x)
            ys.append(y)
        x_src = torch.cat(xs, dim=0).to(device, non_blocking=True)
        y_src = torch.cat(ys, dim=0).to(device, non_blocking=True)
        x_tgt = target_batch[0].to(device, non_blocking=True)  # target labels ignored

        feat_src = backbone(x_src)
        feat_tgt = backbone(x_tgt)
        logits_src = classifier(feat_src)

        cls_loss = CE(logits_src, y_src)
        alignment_loss = rbf_mmd2(feat_src.float(), feat_tgt.float())
        total = cls_loss + lambda_mmd * alignment_loss
        return {"total": total, "cls_loss": cls_loss, "mmd_loss": alignment_loss}
    return compute_losses


def train(backbone, classifier, source_loaders, target_loader, val_loaders,
          steps_per_epoch, results_dir, lambda_mmd=LAMBDA_MMD, device="cuda",
          method_name="dan", **kwargs):
    return run_training(
        backbone, classifier, aux_module=None,
        compute_losses=make_compute_losses(lambda_mmd),
        source_loaders=source_loaders, target_loader=target_loader,
        val_loaders=val_loaders, method_name=method_name,
        results_dir=results_dir, steps_per_epoch=steps_per_epoch,
        device=device, **kwargs)
