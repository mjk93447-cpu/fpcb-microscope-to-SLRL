from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from training.dataset import FieldJoint3ClassDataset, PublicLeadDataset


def test_public_lead_dataset_pairing(tmp_path: Path):
    root = tmp_path / "pub"
    (root / "images").mkdir(parents=True)
    (root / "labels" / "lead_masks").mkdir(parents=True)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "images" / "a.png"), img)
    lm = np.zeros((16, 16), dtype=np.uint8)
    lm[4:12, 4:12] = 255
    cv2.imwrite(str(root / "labels" / "lead_masks" / "a.png"), lm)
    ds = PublicLeadDataset(root, image_size=8, split="train", val_fraction=0.0, seed=0)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (3, 8, 8)
    assert y.shape == (8, 8)
    assert int(y.max()) <= 1


def test_joint_three_class_priority_crack_over_lead(tmp_path: Path):
    pr = tmp_path / "proj"
    (pr / "images").mkdir(parents=True)
    (pr / "labels" / "masks").mkdir(parents=True)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(pr / "images" / "x.png"), img)
    crack = np.zeros((16, 16), dtype=np.uint8)
    lead = np.zeros((16, 16), dtype=np.uint8)
    crack[6:10, 6:10] = 255
    lead[8:12, 8:12] = 255
    cv2.imwrite(str(pr / "labels" / "masks" / "x.png"), crack)
    cv2.imwrite(str(pr / "labels" / "masks" / "x_lead.png"), lead)
    ds = FieldJoint3ClassDataset(pr, image_size=16, split="train", val_fraction=0.0, seed=0)
    assert len(ds) == 1
    _, y = ds[0]
    assert y[7, 7] == 2
    overlap = (crack > 127) & (lead > 127)
    y_np = y.numpy()
    assert int(y_np[overlap].min()) == 2
