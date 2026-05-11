from __future__ import annotations

from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from postprocess import Segment


def generate_heatmap(
    image_shape: Tuple[int, int],
    segments: Iterable[Segment],
    lead_weight: float,
    crack_weight: float,
    blur_kernel: int,
) -> np.ndarray:
    h, w = image_shape
    canvas = np.zeros((h, w), dtype=np.float32)

    for seg in segments:
        if len(seg.points) < 2:
            continue
        weight = crack_weight if seg.label == "crack" else lead_weight
        for i in range(len(seg.points) - 1):
            p1 = seg.points[i]
            p2 = seg.points[i + 1]
            cv2.line(canvas, p1, p2, color=float(weight), thickness=2, lineType=cv2.LINE_AA)

    blur_size = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(canvas, (blur_size, blur_size), sigmaX=0)

    if np.max(blurred) <= 1e-8:
        return np.zeros_like(blurred, dtype=np.uint8)

    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def colorize_heatmap(heatmap_gray: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)


def draw_segments_overlay(image_bgr: np.ndarray, segments: Iterable[Segment]) -> np.ndarray:
    out = image_bgr.copy()
    color_map: Dict[str, Tuple[int, int, int]] = {
        "lead": (0, 255, 0),
        "crack": (0, 0, 255),
    }
    for seg in segments:
        color = color_map.get(seg.label, (255, 255, 255))
        if len(seg.points) == 1:
            cv2.circle(out, seg.points[0], 2, color, -1)
            continue
        for i in range(len(seg.points) - 1):
            cv2.line(out, seg.points[i], seg.points[i + 1], color, 1, cv2.LINE_AA)
    return out

