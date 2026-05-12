from __future__ import annotations

import torch.nn as nn


def freeze_deeplab_backbone_and_aux(model: nn.Module) -> None:
    """Keep main `classifier` trainable; freeze ResNet backbone and aux head."""
    if hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False
    if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        for p in model.aux_classifier.parameters():
            p.requires_grad = False
