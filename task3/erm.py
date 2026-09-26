
import os
import torch

from task3.models.backbone import build_backbone, FEATURE_DIM
from task3.models.classifier_head import build_classifier
from shared.pacs import PACS_CLASSES

N_CLASSES = len(PACS_CLASSES)


def load_erm_checkpoint(shared_checkpoint_dir: str, device: str = "cuda"):
    backbone = build_backbone(device=device)
    classifier = build_classifier(FEATURE_DIM, N_CLASSES, device=device)

    backbone.load_state_dict(torch.load(
        os.path.join(shared_checkpoint_dir, "source_only_backbone.pt"), map_location=device))
    classifier.load_state_dict(torch.load(
        os.path.join(shared_checkpoint_dir, "source_only_classifier.pt"), map_location=device))
    backbone.to(device)
    classifier.to(device)
    return backbone, classifier
