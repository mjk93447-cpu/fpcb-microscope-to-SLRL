from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from labeling.label_io import load_lead_meta_if_any, load_meta_if_any, save_label
from project_layout import LabelMeta


def test_label_roundtrip(tmp_path: Path):
    pr = tmp_path / "proj"
    (pr / "images").mkdir(parents=True)
    img_path = pr / "images" / "a.png"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    mask_path = pr / "labels" / "masks" / "a.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    meta = LabelMeta.new(
        image_path=img_path,
        project_root=pr,
        mask_png_path=mask_path,
        classes=["crack"],
        tool={"name": "test", "version": "0"},
        grabcut={"roi": [1, 2, 10, 10]},
    )
    save_label(pr, img_path, mask, meta)
    m2 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert m2 is not None
    assert np.array_equal(m2, mask)
    meta2 = load_meta_if_any(pr, img_path)
    assert meta2 is not None
    assert meta2.image_filename == "a.png"
    data = json.loads(meta2.to_json())
    assert data["schema_version"] == meta.schema_version


def test_lead_label_roundtrip(tmp_path: Path):
    pr = tmp_path / "proj"
    (pr / "images").mkdir(parents=True)
    img_path = pr / "images" / "b.png"
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    mask_path = pr / "labels" / "masks" / "b_lead.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    meta = LabelMeta.new(
        image_path=img_path,
        project_root=pr,
        mask_png_path=mask_path,
        classes=["lead"],
        tool={"name": "test", "version": "0"},
        grabcut=None,
    )
    save_label(pr, img_path, mask, meta, label_kind="lead")
    m2 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert m2 is not None
    assert np.array_equal(m2, mask)
    meta2 = load_lead_meta_if_any(pr, img_path)
    assert meta2 is not None
    assert meta2.classes == ["lead"]
