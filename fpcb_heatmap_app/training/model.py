from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


def build_deeplab_two_class() -> nn.Module:
    """Same head shape as `inference.SegmentationInference` (2 logits: background, crack)."""
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    return model
