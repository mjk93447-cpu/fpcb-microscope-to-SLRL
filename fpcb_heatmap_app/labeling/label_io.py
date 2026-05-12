from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from project_layout import LabelMeta, label_meta_from_json, label_meta_path


def mask_path_for_image(project_root: Path, image_path: Path) -> Path:
    return project_root / "labels" / "masks" / f"{image_path.stem}.png"


def draft_mask_path(project_root: Path, image_path: Path) -> Path:
    return project_root / "labels" / "masks" / f"{image_path.stem}_draft.png"


def save_label(
    project_root: Path,
    image_path: Path,
    mask_binary_255: np.ndarray,
    meta: LabelMeta,
) -> None:
    mask_path = mask_path_for_image(project_root, image_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    meta.touch_updated()
    meta_path = label_meta_path(project_root, image_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(mask_path), mask_binary_255)
    if not ok:
        raise IOError(f"Failed to write mask: {mask_path}")
    meta_path.write_text(meta.to_json(), encoding="utf-8")


def save_draft_mask(project_root: Path, image_path: Path, mask_binary_255: np.ndarray) -> None:
    p = draft_mask_path(project_root, image_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), mask_binary_255)
    if not ok:
        raise IOError(f"Failed to write draft mask: {p}")


def load_mask_if_any(project_root: Path, image_path: Path) -> np.ndarray | None:
    """Load saved label, or draft, or None."""
    for p in (mask_path_for_image(project_root, image_path), draft_mask_path(project_root, image_path)):
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return m
    return None


def load_meta_if_any(project_root: Path, image_path: Path) -> LabelMeta | None:
    mp = label_meta_path(project_root, image_path)
    if not mp.exists():
        return None
    return label_meta_from_json(mp.read_text(encoding="utf-8"))
