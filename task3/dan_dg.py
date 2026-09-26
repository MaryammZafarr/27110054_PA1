
import torch
import torch.nn as nn

from shared.pacs_protocol import run_training
from shared.mmd import pairwise_domain_mmd

CE = nn.CrossEntropyLoss()


def make_compute_losses(lambda_dg: float = 1.0):
    def compute_losses(backbone, classifier, aux_module, source_batches, target_batch, p, device):
        feats_by_domain, all_logits, all_labels = {}, [], []
        for domain, (x, y) in source_batches.items():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feat = backbone(x)
            feats_by_domain[domain] = feat
            all_logits.append(classifier(feat))
            all_labels.append(y)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        cls_loss = CE(logits, labels)

        mmd_loss = pairwise_domain_mmd(feats_by_domain)
        total = cls_loss + lambda_dg * mmd_loss

        return {"total": total, "cls_loss": cls_loss, "mmd_loss": mmd_loss}
    return compute_losses


def train(backbone, classifier, source_loaders, val_loaders, steps_per_epoch,
          results_dir, lambda_dg=1.0, device="cuda", method_name="dan_dg", **kwargs):
    return run_training(
        backbone, classifier, aux_module=None,
        compute_losses=make_compute_losses(lambda_dg),
        source_loaders=source_loaders, target_loader=None,
        val_loaders=val_loaders, method_name=method_name,
        results_dir=results_dir, steps_per_epoch=steps_per_epoch,
        device=device, **kwargs)
