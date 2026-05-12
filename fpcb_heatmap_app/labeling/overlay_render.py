from __future__ import annotations

import numpy as np


def blend_mask_bgr(
    image_bgr: np.ndarray,
    mask_255: np.ndarray,
    *,
    color_bgr: tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    if mask_255.shape[:2] != image_bgr.shape[:2]:
        raise ValueError("mask and image shape mismatch")
    out = image_bgr.astype(np.float32).copy()
    m = (mask_255 > 127).astype(np.float32)[..., None]
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    overlay = out * (1.0 - m * alpha) + color * (m * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)
