from __future__ import annotations

"""
Pre-train the 3-class DeepLab head on **public** copper-lead imagery before field deployment.

Expected dataset layout::

    <public_root>/
      images/<name>.png
      labels/lead_masks/<name>.png   # 255 = lead foreground

Only classes 0 (background) and 1 (lead) appear in targets; class 2 (crack) logits stay trainable
via the softmax but receive no positive pixels in this phase — field `train.py --mode joint`
or crack labels later specialize them.

Example::

    python pretrain_lead_public.py --public-root D:/datasets/public_lead_bundle --epochs 20
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.dataset import PublicLeadDataset
from training.model import build_deeplab_three_class


def dice_lead(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    return _dice_class(logits, target, class_idx=1, eps=eps)


def _dice_class(logits: torch.Tensor, target: torch.Tensor, class_idx: int, eps: float = 1e-6) -> float:
    prob = torch.softmax(logits, dim=1)[:, class_idx]
    pred = (prob > 0.5).float()
    t = (target == class_idx).float()
    inter = (pred * t).sum()
    union = pred.sum() + t.sum()
    return float((2 * inter + eps) / (union + eps))


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-train 3-class DeepLab on public lead masks only.")
    ap.add_argument("--public-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="Output .pt path (default: <public_root>/checkpoints/lead_public_3c_best.pt)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = args.public_root.resolve()
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out if args.out else ckpt_dir / "lead_public_3c_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = PublicLeadDataset(root, image_size=args.image_size, split="train", seed=args.seed)
    val_ds = PublicLeadDataset(root, image_size=args.image_size, split="val", seed=args.seed)
    if len(train_ds) == 0:
        raise SystemExit(
            f"No pairs under {root}: need images/ + labels/lead_masks/ with matching stems (.png)."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_deeplab_three_class().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = -1.0
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
                    dices.append(dice_lead(logits[i : i + 1], y[i : i + 1]))
        mean_dice = float(sum(dices) / max(1, len(dices)))
        print(f"epoch={epoch+1}/{args.epochs} train_loss={avg_loss:.4f} val_lead_dice={mean_dice:.4f}")

        if mean_dice >= best:
            best = mean_dice
            torch.save(
                {"state_dict": model.state_dict(), "meta": {"val_lead_dice": mean_dice, "num_classes": 3}},
                out_path,
            )
            print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
