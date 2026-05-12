from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from io_utils import IMAGE_EXTENSIONS
from labeling.label_io import crack_mask_path, lead_mask_path, mask_path_for_image


def _split_pairs(
    pairs: list[tuple[Path, ...]],
    *,
    split: str,
    val_fraction: float,
    seed: int,
) -> list[tuple[Path, ...]]:
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if len(pairs) == 0:
        return []
    if len(pairs) == 1:
        return pairs
    n_val = max(1, int(len(pairs) * val_fraction))
    n_val = min(n_val, len(pairs) - 1)
    if split == "train":
        return pairs[n_val:]
    return pairs[:n_val]


class CrackMaskDataset(Dataset):
    """Pairs `images/*` with `labels/masks/<stem>.png` (255 = crack foreground)."""

    def __init__(
        self,
        project_root: Path,
        *,
        image_size: int = 256,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.project_root = project_root
        self.image_size = image_size
        self.split = split
        imgs_dir = project_root / "images"
        pairs: list[tuple[Path, Path]] = []
        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mp = mask_path_for_image(project_root, img_path)
            if not mp.exists():
                continue
            pairs.append((img_path, mp))
        self.pairs = _split_pairs(pairs, split=split, val_fraction=val_fraction, seed=seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.pairs[idx]
        bgr = cv2.imread(str(ip))
        if bgr is None:
            raise ValueError(f"Could not read image: {ip}")
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mp}")
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        y = (mask > 127).astype(np.int64)
        x = rgb.astype(np.float32) / 255.0
        xt = torch.from_numpy(x).permute(2, 0, 1)
        yt = torch.from_numpy(y)
        return xt, yt


class PublicLeadDataset(Dataset):
    """
    Public bundle for pre-training lead only:
      <root>/images/<name>.png
      <root>/labels/lead_masks/<name>.png   (255 = lead)
    Target values are 0 = background, 1 = lead (compatible with a 3-class head; crack class unused).
    """

    def __init__(
        self,
        public_root: Path,
        *,
        image_size: int = 256,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.public_root = public_root
        self.image_size = image_size
        imgs_dir = public_root / "images"
        lead_dir = public_root / "labels" / "lead_masks"
        pairs: list[tuple[Path, Path]] = []
        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            lp = lead_dir / f"{img_path.stem}.png"
            if not lp.exists():
                continue
            pairs.append((img_path, lp))
        self.pairs = _split_pairs(pairs, split=split, val_fraction=val_fraction, seed=seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, lp = self.pairs[idx]
        bgr = cv2.imread(str(ip))
        if bgr is None:
            raise ValueError(f"Could not read image: {ip}")
        lmask = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        if lmask is None:
            raise ValueError(f"Could not read lead mask: {lp}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        lmask = cv2.resize(lmask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        y = (lmask > 127).astype(np.int64)
        x = rgb.astype(np.float32) / 255.0
        xt = torch.from_numpy(x).permute(2, 0, 1)
        yt = torch.from_numpy(y)
        return xt, yt


class FieldLeadCalibrationDataset(Dataset):
    """Field images under `images/` with `labels/masks/<stem>_lead.png` (255 = lead). Same y semantics as `PublicLeadDataset`."""

    def __init__(
        self,
        project_root: Path,
        *,
        image_size: int = 256,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.project_root = project_root
        self.image_size = image_size
        imgs_dir = project_root / "images"
        pairs: list[tuple[Path, Path]] = []
        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            lp = lead_mask_path(project_root, img_path)
            if not lp.exists():
                continue
            pairs.append((img_path, lp))
        self.pairs = _split_pairs(pairs, split=split, val_fraction=val_fraction, seed=seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, lp = self.pairs[idx]
        bgr = cv2.imread(str(ip))
        if bgr is None:
            raise ValueError(f"Could not read image: {ip}")
        lmask = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        if lmask is None:
            raise ValueError(f"Could not read lead mask: {lp}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        lmask = cv2.resize(lmask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        y = (lmask > 127).astype(np.int64)
        x = rgb.astype(np.float32) / 255.0
        xt = torch.from_numpy(x).permute(2, 0, 1)
        yt = torch.from_numpy(y)
        return xt, yt


class FieldJoint3ClassDataset(Dataset):
    """
    Requires crack mask `labels/masks/<stem>.png`.
    Optional lead mask `labels/masks/<stem>_lead.png`; if missing, only background vs crack is labeled.
    Pixel classes: 0 = background, 1 = lead, 2 = crack. Crack overrides lead on overlap.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        image_size: int = 256,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.project_root = project_root
        self.image_size = image_size
        imgs_dir = project_root / "images"
        pairs: list[tuple[Path, Path, Path | None]] = []
        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            cp = crack_mask_path(project_root, img_path)
            if not cp.exists():
                continue
            lp = lead_mask_path(project_root, img_path)
            pairs.append((img_path, cp, lp if lp.exists() else None))
        self.pairs = _split_pairs(pairs, split=split, val_fraction=val_fraction, seed=seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, cp, lp_opt = self.pairs[idx]
        bgr = cv2.imread(str(ip))
        if bgr is None:
            raise ValueError(f"Could not read image: {ip}")
        cmask = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
        if cmask is None:
            raise ValueError(f"Could not read crack mask: {cp}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        cmask = cv2.resize(cmask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        crack_fg = cmask > 127
        lead_fg = np.zeros_like(crack_fg, dtype=bool)
        if lp_opt is not None:
            lmask = cv2.imread(str(lp_opt), cv2.IMREAD_GRAYSCALE)
            if lmask is not None:
                lmask = cv2.resize(lmask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                lead_fg = lmask > 127
        y = np.zeros(crack_fg.shape, dtype=np.int64)
        y[lead_fg & ~crack_fg] = 1
        y[crack_fg] = 2
        x = rgb.astype(np.float32) / 255.0
        xt = torch.from_numpy(x).permute(2, 0, 1)
        yt = torch.from_numpy(y)
        return xt, yt
