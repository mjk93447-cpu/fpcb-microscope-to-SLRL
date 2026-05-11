from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from config import AppConfig
from heatmap import colorize_heatmap, draw_segments_overlay, generate_heatmap
from inference import SegmentationInference
from io_utils import append_summary_csv, save_image, save_segments_json
from postprocess import collect_segments


@dataclass
class ProcessResult:
    image_name: str
    lead_segments: int
    crack_segments: int
    total_segments: int
    lead_score: float
    crack_score: float
    overlay_path: Path
    heatmap_path: Path
    json_path: Path


class FpcbProcessor:
    def __init__(self, config: AppConfig, checkpoint_path: str | None = None):
        self.config = config
        self.infer = SegmentationInference(
            checkpoint_path=checkpoint_path,
            confidence_threshold=config.confidence_threshold,
            use_gpu=config.use_gpu,
            use_torch_backbone=config.use_torch_backbone,
        )

    def process_image(self, image_path: Path, output_dir: Path) -> ProcessResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        inf = self.infer.predict(image)
        segments = collect_segments(
            lead_mask=inf.lead_mask,
            crack_mask=inf.crack_mask,
            min_area=self.config.min_component_area,
            min_length=self.config.min_segment_length,
            lead_score=inf.lead_score,
            crack_score=inf.crack_score,
        )

        heatmap_gray = generate_heatmap(
            image_shape=image.shape[:2],
            segments=segments,
            lead_weight=self.config.lead_weight,
            crack_weight=self.config.crack_weight,
            blur_kernel=self.config.heatmap_blur_kernel,
        )
        heatmap_color = colorize_heatmap(heatmap_gray)
        overlay = draw_segments_overlay(image, segments)

        stem = image_path.stem
        overlay_path = output_dir / f"{stem}_overlay.png"
        heatmap_path = output_dir / f"{stem}_heatmap.png"
        json_path = output_dir / f"{stem}_segments.json"
        summary_path = output_dir / "summary.csv"

        save_image(overlay_path, overlay)
        save_image(heatmap_path, heatmap_color)
        save_segments_json(
            json_path,
            segments,
            meta={
                "image": image_path.name,
                "engine": self.infer.info(),
                "lead_score": inf.lead_score,
                "crack_score": inf.crack_score,
            },
        )

        lead_count = sum(1 for s in segments if s.label == "lead")
        crack_count = sum(1 for s in segments if s.label == "crack")
        row = {
            "image": image_path.name,
            "lead_segments": lead_count,
            "crack_segments": crack_count,
            "total_segments": len(segments),
            "lead_score": f"{inf.lead_score:.6f}",
            "crack_score": f"{inf.crack_score:.6f}",
        }
        append_summary_csv(summary_path, row)

        return ProcessResult(
            image_name=image_path.name,
            lead_segments=lead_count,
            crack_segments=crack_count,
            total_segments=len(segments),
            lead_score=inf.lead_score,
            crack_score=inf.crack_score,
            overlay_path=overlay_path,
            heatmap_path=heatmap_path,
            json_path=json_path,
        )

