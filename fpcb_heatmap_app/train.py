from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from project_layout import build_project_paths, ensure_project_dirs
from training.dataset import CrackMaskDataset, FieldJoint3ClassDataset, FieldLeadCalibrationDataset
from training.freeze import freeze_deeplab_backbone_and_aux
from training.model import build_deeplab, build_deeplab_three_class, build_deeplab_two_class


def dice_binary_from_logits(logits: torch.Tensor, target: torch.Tensor, class_idx: int, eps: float = 1e-6) -> float:
    prob = torch.softmax(logits, dim=1)[:, class_idx]
    pred = (prob > 0.5).float()
    t = (target == class_idx).float()
    inter = (pred * t).sum()
    union = pred.sum() + t.sum()
    return float((2 * inter + eps) / (union + eps))


def dice_crack_legacy(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    prob = torch.softmax(logits, dim=1)[:, 1]
    pred = (prob > 0.5).float()
    t = target.float()
    inter = (pred * t).sum()
    union = pred.sum() + t.sum()
    return float((2 * inter + eps) / (union + eps))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train segmentation: crack-only, joint 3-class, or on-site lead calibration.")
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument(
        "--mode",
        choices=("crack", "joint", "calibrate-lead"),
        default="joint",
        help="crack=2-class legacy; joint=bg/lead/crack; calibrate-lead=load --init-checkpoint, freeze backbone, tune on field lead masks.",
    )
    ap.add_argument("--init-checkpoint", type=Path, default=None, help="Required for calibrate-lead (e.g. public pretrain .pt).")
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

    if args.mode == "crack":
        train_ds = CrackMaskDataset(pr, image_size=args.image_size, split="train", seed=args.seed)
        val_ds = CrackMaskDataset(pr, image_size=args.image_size, split="val", seed=args.seed)
        if len(train_ds) == 0:
            raise SystemExit("No labeled pairs found (need images/ + labels/masks/<stem>.png crack masks).")
        model = build_deeplab_two_class().to(device)
        out_ckpt = pp.checkpoints_dir / "crack_deeplab_best.pt"
        num_classes = 2

        def batch_dice(logits: torch.Tensor, y: torch.Tensor) -> float:
            return dice_crack_legacy(logits, y)

    elif args.mode == "joint":
        train_ds = FieldJoint3ClassDataset(pr, image_size=args.image_size, split="train", seed=args.seed)
        val_ds = FieldJoint3ClassDataset(pr, image_size=args.image_size, split="val", seed=args.seed)
        if len(train_ds) == 0:
            raise SystemExit("No joint training pairs (need images/ + labels/masks/<stem>.png for crack).")
        model = build_deeplab_three_class().to(device)
        out_ckpt = pp.checkpoints_dir / "segmentation_deeplab_3c_best.pt"
        num_classes = 3

        def batch_dice(logits: torch.Tensor, y: torch.Tensor) -> float:
            d1 = dice_binary_from_logits(logits, y, 1)
            d2 = dice_binary_from_logits(logits, y, 2)
            return (d1 + d2) * 0.5

    else:
        if not args.init_checkpoint or not args.init_checkpoint.is_file():
            raise SystemExit("calibrate-lead requires --init-checkpoint pointing to an existing .pt file.")
        train_ds = FieldLeadCalibrationDataset(pr, image_size=args.image_size, split="train", seed=args.seed)
        val_ds = FieldLeadCalibrationDataset(pr, image_size=args.image_size, split="val", seed=args.seed)
        if len(train_ds) == 0:
            raise SystemExit("No lead calibration pairs (need images/ + labels/masks/<stem>_lead.png).")
        ck = torch.load(args.init_checkpoint, map_location="cpu")
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
        nc = int(meta.get("num_classes", 3)) if isinstance(meta, dict) else 3
        if nc != 3:
            raise SystemExit("calibrate-lead expects a 3-class init checkpoint (num_classes=3).")
        model = build_deeplab(3).to(device)
        model.load_state_dict(sd, strict=False)
        freeze_deeplab_backbone_and_aux(model)
        out_ckpt = pp.checkpoints_dir / "lead_calibrated_3c_best.pt"
        num_classes = 3

        def batch_dice(logits: torch.Tensor, y: torch.Tensor) -> float:
            return dice_binary_from_logits(logits, y, 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)

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
                    dices.append(batch_dice(logits[i : i + 1], y[i : i + 1]))
        mean_dice = float(sum(dices) / max(1, len(dices)))
        print(f"epoch={epoch+1}/{args.epochs} train_loss={avg_loss:.4f} val_dice={mean_dice:.4f}")

        if mean_dice >= best:
            best = mean_dice
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "meta": {"val_dice": mean_dice, "num_classes": num_classes, "mode": args.mode},
                },
                out_ckpt,
            )
            print(f"  saved {out_ckpt}")


if __name__ == "__main__":
    main()
