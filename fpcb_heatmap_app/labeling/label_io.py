from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from project_layout import LabelMeta, label_meta_from_json, label_meta_path, label_meta_path_lead

LabelKind = Literal["crack", "lead"]


def crack_mask_path(project_root: Path, image_path: Path) -> Path:
    return project_root / "labels" / "masks" / f"{image_path.stem}.png"


def lead_mask_path(project_root: Path, image_path: Path) -> Path:
    return project_root / "labels" / "masks" / f"{image_path.stem}_lead.png"


def mask_path_for_image(project_root: Path, image_path: Path) -> Path:
    """Crack mask path (legacy name). Same as `crack_mask_path`."""
    return crack_mask_path(project_root, image_path)


def draft_crack_mask_path(project_root: Path, image_path: Path) -> Path:
    return project_root / "labels" / "masks" / f"{image_path.stem}_draft.png"


def draft_lead_mask_path(project_root: Path, image_path: Path) -> Path:
    return project_root / "labels" / "masks" / f"{image_path.stem}_lead_draft.png"


def draft_mask_path(project_root: Path, image_path: Path) -> Path:
    """Crack draft (legacy). Same as `draft_crack_mask_path`."""
    return draft_crack_mask_path(project_root, image_path)


def _mask_path_for_kind(project_root: Path, image_path: Path, kind: LabelKind) -> Path:
    return crack_mask_path(project_root, image_path) if kind == "crack" else lead_mask_path(project_root, image_path)


def _draft_path_for_kind(project_root: Path, image_path: Path, kind: LabelKind) -> Path:
    return draft_crack_mask_path(project_root, image_path) if kind == "crack" else draft_lead_mask_path(
        project_root, image_path
    )


def _meta_path_for_kind(project_root: Path, image_path: Path, kind: LabelKind) -> Path:
    return label_meta_path(project_root, image_path) if kind == "crack" else label_meta_path_lead(
        project_root, image_path
    )


def save_label(
    project_root: Path,
    image_path: Path,
    mask_binary_255: np.ndarray,
    meta: LabelMeta,
    *,
    label_kind: LabelKind = "crack",
) -> None:
    mask_path = _mask_path_for_kind(project_root, image_path, label_kind)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    meta.touch_updated()
    meta_path = _meta_path_for_kind(project_root, image_path, label_kind)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(mask_path), mask_binary_255)
    if not ok:
        raise IOError(f"Failed to write mask: {mask_path}")
    meta_path.write_text(meta.to_json(), encoding="utf-8")


def save_draft_mask(
    project_root: Path,
    image_path: Path,
    mask_binary_255: np.ndarray,
    *,
    label_kind: LabelKind = "crack",
) -> None:
    p = _draft_path_for_kind(project_root, image_path, label_kind)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), mask_binary_255)
    if not ok:
        raise IOError(f"Failed to write draft mask: {p}")


def load_mask_if_any(
    project_root: Path,
    image_path: Path,
    *,
    label_kind: LabelKind = "crack",
) -> np.ndarray | None:
    """Load saved label, or draft, or None."""
    for p in (_mask_path_for_kind(project_root, image_path, label_kind), _draft_path_for_kind(project_root, image_path, label_kind)):
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


def load_lead_meta_if_any(project_root: Path, image_path: Path) -> LabelMeta | None:
    mp = label_meta_path_lead(project_root, image_path)
    if not mp.exists():
        return None
    return label_meta_from_json(mp.read_text(encoding="utf-8"))
