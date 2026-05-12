from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import AppConfig
from pipeline import FpcbProcessor
from project_layout import build_project_paths, ensure_project_dirs


def test_nested_outputs_layout(tmp_path: Path):
    pr = tmp_path / "proj"
    pp = build_project_paths(pr)
    ensure_project_dirs(pp)
    (pp.images_dir / "x.png").parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(pp.images_dir / "x.png"), img)

    cfg = AppConfig(use_gpu=False, use_torch_backbone=False, nested_project_layout=True)
    proc = FpcbProcessor(cfg, checkpoint_path=None)
    r = proc.process_image(pp.images_dir / "x.png", pp.outputs_dir)

    assert r.overlay_path.parent == pp.outputs_overlays_dir
    assert r.heatmap_path.parent == pp.outputs_heatmaps_dir
    assert r.json_path.parent == pp.outputs_segments_dir
    assert pp.outputs_summary_csv.exists()
