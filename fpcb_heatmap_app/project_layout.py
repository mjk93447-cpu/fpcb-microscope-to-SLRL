from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional


LABEL_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    images_dir: Path
    labels_masks_dir: Path
    labels_meta_dir: Path
    outputs_dir: Path
    outputs_overlays_dir: Path
    outputs_heatmaps_dir: Path
    outputs_segments_dir: Path
    outputs_summary_csv: Path
    checkpoints_dir: Path
    runs_dir: Path


def build_project_paths(project_root: Path) -> ProjectPaths:
    root = project_root
    return ProjectPaths(
        root=root,
        images_dir=root / "images",
        labels_masks_dir=root / "labels" / "masks",
        labels_meta_dir=root / "labels" / "meta",
        outputs_dir=root / "outputs",
        outputs_overlays_dir=root / "outputs" / "overlays",
        outputs_heatmaps_dir=root / "outputs" / "heatmaps",
        outputs_segments_dir=root / "outputs" / "segments",
        outputs_summary_csv=root / "outputs" / "summary.csv",
        checkpoints_dir=root / "checkpoints",
        runs_dir=root / "runs",
    )


def ensure_project_dirs(p: ProjectPaths) -> None:
    for d in [
        p.images_dir,
        p.labels_masks_dir,
        p.labels_meta_dir,
        p.outputs_overlays_dir,
        p.outputs_heatmaps_dir,
        p.outputs_segments_dir,
        p.checkpoints_dir,
        p.runs_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


LabelClass = Literal["crack", "lead"]


@dataclass
class LabelMeta:
    schema_version: str
    image_filename: str
    image_sha256: str
    mask_png_relpath: str
    created_at: str
    updated_at: str
    classes: list[LabelClass]
    tool: dict
    grabcut: Optional[dict] = None

    @staticmethod
    def new(
        *,
        image_path: Path,
        project_root: Path,
        mask_png_path: Path,
        classes: list[LabelClass],
        tool: dict,
        grabcut: Optional[dict] = None,
    ) -> "LabelMeta":
        now = datetime.now(timezone.utc).isoformat()
        return LabelMeta(
            schema_version=LABEL_SCHEMA_VERSION,
            image_filename=image_path.name,
            image_sha256=sha256_file(image_path),
            mask_png_relpath=str(mask_png_path.relative_to(project_root)).replace("\\", "/"),
            created_at=now,
            updated_at=now,
            classes=classes,
            tool=tool,
            grabcut=grabcut,
        )

    def touch_updated(self) -> None:
        object.__setattr__(self, "updated_at", datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def label_meta_from_json(text: str) -> LabelMeta:
    data = json.loads(text)
    return LabelMeta(**data)


def label_meta_path(project_root: Path, image_path: Path) -> Path:
    # Store meta by stem; collisions are possible for duplicate stems in different folders,
    # but Phase 1 assumes one canonical images/ folder to avoid ambiguity.
    return project_root / "labels" / "meta" / f"{image_path.stem}.json"

