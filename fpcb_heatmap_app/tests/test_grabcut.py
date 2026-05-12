from __future__ import annotations

import cv2
import numpy as np

from labeling.grabcut_engine import gc_mask_to_binary, init_grabcut


def test_grabcut_runs_on_synthetic():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[40:80, 60:100] = (200, 200, 200)
    rect = (50, 30, 60, 60)
    st = init_grabcut(img, rect, iters=2)
    bin_mask = gc_mask_to_binary(st.mask_gc)
    assert bin_mask.shape == (120, 160)
    assert bin_mask.dtype == np.uint8
