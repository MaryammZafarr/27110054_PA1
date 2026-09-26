
import torch
import torch.nn as nn

from task2.models.domain_discriminator import grad_reverse, grl_alpha_schedule
from shared.pacs_protocol import run_training

CE = nn.CrossEntropyLoss()


def make_compute_losses(max_alpha: float = 1.0):
    def compute_losses(backbone, classifier, aux_module, source_batches, target_batch, p, device):
        xs, ys = [], []
        for _, (x, y) in source_batches.items():
            xs.append(x)
            ys.append(y)
        x_src = torch.cat(xs, dim=0).to(device, non_blocking=True)
        y_src = torch.cat(ys, dim=0).to(device, non_blocking=True)
        x_tgt = target_batch[0].to(device, non_blocking=True)

        feat_src = backbone(x_src)
        feat_tgt = backbone(x_tgt)
        logits_src = classifier(feat_src)
        cls_loss = CE(logits_src, y_src)

        alpha = grl_alpha_schedule(p, max_alpha=max_alpha)
        feat_all = torch.cat([feat_src, feat_tgt], dim=0)
        domain_labels = torch.cat([
            torch.zeros(feat_src.shape[0], dtype=torch.long),
            torch.ones(feat_tgt.shape[0], dtype=torch.long),
        ]).to(device, non_blocking=True)

        reversed_feat = grad_reverse(feat_all, alpha)
        domain_logits = aux_module(reversed_feat)
        domain_loss = CE(domain_logits, domain_labels)

        total = cls_loss + domain_loss
        with torch.no_grad():
            domain_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean()
            feat_norm = feat_all.norm(dim=1).mean()

        return {"total": total, "cls_loss": cls_loss, "domain_loss": domain_loss,
                "domain_acc": domain_acc, "alpha": alpha,"feat_norm": feat_norm}
    return compute_losses


def train(backbone, classifier, discriminator, source_loaders, target_loader,
          val_loaders, steps_per_epoch, results_dir, max_alpha=1.0, device="cuda",
          method_name="dann", **kwargs):
    return run_training(
        backbone, classifier, aux_module=discriminator,
        compute_losses=make_compute_losses(max_alpha),
        source_loaders=source_loaders, target_loader=target_loader,
        val_loaders=val_loaders, method_name=method_name,
        results_dir=results_dir, steps_per_epoch=steps_per_epoch,
        device=device, **kwargs)