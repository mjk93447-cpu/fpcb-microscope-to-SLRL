from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.dataset import CrackMaskDataset
from training.model import build_deeplab_two_class


def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    prob = torch.softmax(logits, dim=1)[:, 1]
    pred = (prob > 0.5).float()
    t = target.float()
    inter = (pred * t).sum()
    union = pred.sum() + t.sum()
    return float((2 * inter + eps) / (union + eps))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate crack segmentation checkpoint.")
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CrackMaskDataset(args.project_root, image_size=args.image_size, split="val", seed=args.seed)
    if len(ds) == 0:
        ds = CrackMaskDataset(args.project_root, image_size=args.image_size, split="train", seed=args.seed)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    model = build_deeplab_two_class().to(device)
    ck = torch.load(args.checkpoint, map_location="cpu")
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.load_state_dict(state, strict=False)
    model.eval()

    dices: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)["out"]
            for i in range(x.size(0)):
                dices.append(dice_score(logits[i : i + 1], y[i : i + 1]))
    print(f"mean_dice={sum(dices)/max(1,len(dices)):.4f} n={len(dices)}")


if __name__ == "__main__":
    main()
