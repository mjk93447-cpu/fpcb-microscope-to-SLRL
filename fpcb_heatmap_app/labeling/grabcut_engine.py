from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class GrabCutState:
    rect: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    bgd_model: Optional[np.ndarray] = None
    fgd_model: Optional[np.ndarray] = None
    mask_gc: Optional[np.ndarray] = None  # values in {cv2.GC_*}


def init_grabcut(image_bgr: np.ndarray, rect_xywh: Tuple[int, int, int, int], iters: int = 2) -> GrabCutState:
    h, w = image_bgr.shape[:2]
    x, y, rw, rh = rect_xywh
    x = int(max(0, min(x, w - 1)))
    y = int(max(0, min(y, h - 1)))
    rw = int(max(1, min(rw, w - x)))
    rh = int(max(1, min(rh, h - y)))

    mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (x, y, rw, rh)

    cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
    return GrabCutState(rect=rect, bgd_model=bgd_model, fgd_model=fgd_model, mask_gc=mask)


def apply_scribbles(
    state: GrabCutState,
    *,
    fg_scribble: Optional[np.ndarray] = None,
    bg_scribble: Optional[np.ndarray] = None,
) -> None:
    if state.mask_gc is None:
        raise ValueError("GrabCut is not initialized")
    if fg_scribble is not None:
        state.mask_gc[fg_scribble > 0] = cv2.GC_FGD
    if bg_scribble is not None:
        state.mask_gc[bg_scribble > 0] = cv2.GC_BGD


def refine_grabcut(image_bgr: np.ndarray, state: GrabCutState, iters: int = 1) -> GrabCutState:
    if state.mask_gc is None or state.bgd_model is None or state.fgd_model is None:
        raise ValueError("GrabCut is not initialized")

    cv2.grabCut(
        image_bgr,
        state.mask_gc,
        None,
        state.bgd_model,
        state.fgd_model,
        iters,
        cv2.GC_INIT_WITH_MASK,
    )
    return state


def gc_mask_to_binary(mask_gc: np.ndarray) -> np.ndarray:
    # Foreground classes: GC_FGD and GC_PR_FGD
    fg = (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD)
    return (fg.astype(np.uint8) * 255)

