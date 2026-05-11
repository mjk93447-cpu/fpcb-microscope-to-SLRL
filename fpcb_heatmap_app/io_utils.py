from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

import cv2

from postprocess import Segment


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = [x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def save_segments_json(path: Path, segments: Iterable[Segment], meta: dict) -> None:
    payload = {
        "meta": meta,
        "segments": [s.to_dict() for s in segments],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_summary_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "lead_segments",
                "crack_segments",
                "total_segments",
                "lead_score",
                "crack_score",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)

