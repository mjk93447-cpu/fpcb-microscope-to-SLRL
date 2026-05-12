from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from project_layout import build_project_paths, ensure_project_dirs
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
    ap = argparse.ArgumentParser(description="Train crack segmentation (2-class DeepLab head).")
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pr = args.project_root.resolve()
    pp = build_project_paths(pr)
    ensure_project_dirs(pp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CrackMaskDataset(pr, image_size=args.image_size, split="train", seed=args.seed)
    val_ds = CrackMaskDataset(pr, image_size=args.image_size, split="val", seed=args.seed)
    if len(train_ds) == 0:
        raise SystemExit("No labeled pairs found (need images/ + labels/masks/*.png).")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_deeplab_two_class().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = -1.0
    out_ckpt = pp.checkpoints_dir / "crack_deeplab_best.pt"
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)["out"]
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * x.size(0)
        avg_loss = total_loss / max(1, len(train_ds))

        model.eval()
        dices: list[float] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)["out"]
                for i in range(x.size(0)):
                    dices.append(dice_score(logits[i : i + 1], y[i : i + 1]))
        mean_dice = float(sum(dices) / max(1, len(dices)))
        print(f"epoch={epoch+1}/{args.epochs} train_loss={avg_loss:.4f} val_dice={mean_dice:.4f}")

        if mean_dice >= best:
            best = mean_dice
            torch.save({"state_dict": model.state_dict(), "meta": {"val_dice": mean_dice}}, out_ckpt)
            print(f"  saved {out_ckpt}")


if __name__ == "__main__":
    main()
