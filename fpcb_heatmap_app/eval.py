from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.dataset import CrackMaskDataset, FieldJoint3ClassDataset
from training.model import build_deeplab, build_deeplab_two_class


def dice_crack_legacy(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    prob = torch.softmax(logits, dim=1)[:, 1]
    pred = (prob > 0.5).float()
    t = target.float()
    inter = (pred * t).sum()
    union = pred.sum() + t.sum()
    return float((2 * inter + eps) / (union + eps))


def dice_class(logits: torch.Tensor, target: torch.Tensor, class_idx: int, eps: float = 1e-6) -> float:
    prob = torch.softmax(logits, dim=1)[:, class_idx]
    pred = (prob > 0.5).float()
    t = (target == class_idx).float()
    inter = (pred * t).sum()
    union = pred.sum() + t.sum()
    return float((2 * inter + eps) / (union + eps))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate segmentation checkpoint (2-class crack or 3-class joint).")
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--mode", choices=("crack", "joint"), default="joint")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu")
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    nc = int(meta.get("num_classes", 0)) if isinstance(meta, dict) else 0
    if nc not in (2, 3):
        for k, v in state.items():
            if k.endswith("classifier.4.weight") and hasattr(v, "shape"):
                nc = int(v.shape[0])
                break
        if nc not in (2, 3):
            nc = 2

    if args.mode == "crack" or nc == 2:
        ds = CrackMaskDataset(args.project_root, image_size=args.image_size, split="val", seed=args.seed)
        if len(ds) == 0:
            ds = CrackMaskDataset(args.project_root, image_size=args.image_size, split="train", seed=args.seed)
        model = build_deeplab_two_class().to(device)
        model.load_state_dict(state, strict=False)
        model.eval()
        dices: list[float] = []
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)["out"]
                for i in range(x.size(0)):
                    dices.append(dice_crack_legacy(logits[i : i + 1], y[i : i + 1]))
        print(f"mean_crack_dice={sum(dices)/max(1,len(dices)):.4f} n={len(dices)} num_classes=2")
        return

    ds = FieldJoint3ClassDataset(args.project_root, image_size=args.image_size, split="val", seed=args.seed)
    if len(ds) == 0:
        ds = FieldJoint3ClassDataset(args.project_root, image_size=args.image_size, split="train", seed=args.seed)
    model = build_deeplab(3).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    d_lead: list[float] = []
    d_crack: list[float] = []
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)["out"]
            for i in range(x.size(0)):
                d_lead.append(dice_class(logits[i : i + 1], y[i : i + 1], 1))
                d_crack.append(dice_class(logits[i : i + 1], y[i : i + 1], 2))
    print(
        f"mean_lead_dice={sum(d_lead)/max(1,len(d_lead)):.4f} "
        f"mean_crack_dice={sum(d_crack)/max(1,len(d_crack)):.4f} n={len(d_lead)} num_classes=3"
    )


if __name__ == "__main__":
    main()
