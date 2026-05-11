from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from config import AppConfig
from pipeline import FpcbProcessor


def _write_synthetic_image(path: Path) -> None:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.line(img, (20, 120), (240, 120), (255, 255, 255), 2)
    cv2.line(img, (60, 60), (200, 200), (255, 255, 255), 1)
    ok = cv2.imwrite(str(path), img)
    assert ok


def test_outputs_smoke(tmp_path: Path):
    in_path = tmp_path / "sample.png"
    _write_synthetic_image(in_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig(use_gpu=False, use_torch_backbone=False, output_dir=out_dir)
    proc = FpcbProcessor(cfg, checkpoint_path=None)
    result = proc.process_image(in_path, out_dir)

    assert result.overlay_path.exists()
    assert result.heatmap_path.exists()
    assert result.json_path.exists()
    assert (out_dir / "summary.csv").exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert "meta" in payload
    assert "segments" in payload
    assert isinstance(payload["segments"], list)

