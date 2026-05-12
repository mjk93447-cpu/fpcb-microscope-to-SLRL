from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from io_utils import IMAGE_EXTENSIONS
from labeling.label_io import mask_path_for_image


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
        rng = random.Random(seed)
        rng.shuffle(pairs)
        if len(pairs) == 0:
            self.pairs = []
        elif len(pairs) == 1:
            self.pairs = pairs
        else:
            n_val = max(1, int(len(pairs) * val_fraction))
            n_val = min(n_val, len(pairs) - 1)
            if split == "train":
                self.pairs = pairs[n_val:]
            else:
                self.pairs = pairs[:n_val]

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
