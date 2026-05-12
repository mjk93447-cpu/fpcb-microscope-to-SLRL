from __future__ import annotations

import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


def build_deeplab(num_classes: int) -> nn.Module:
    """DeepLabV3-ResNet50 with a linear head of `num_classes` logits (including background)."""
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def build_deeplab_two_class() -> nn.Module:
    """Legacy 2-class head: class 0 = background, class 1 = crack."""
    return build_deeplab(2)


def build_deeplab_three_class() -> nn.Module:
    """3-class: 0 = background, 1 = lead (copper wire), 2 = crack."""
    return build_deeplab(3)
