from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    confidence_threshold: float = 0.30
    min_segment_length: float = 18.0
    min_component_area: int = 8
    heatmap_blur_kernel: int = 21
    crack_weight: float = 2.4
    lead_weight: float = 1.0
    use_gpu: bool = True
    use_torch_backbone: bool = False
    input_size: int = 1024
    output_dir: Path = Path("output")
    # If True, output_dir must be `<project>/outputs`; files go to overlays/, heatmaps/, segments/, summary.csv.
    nested_project_layout: bool = False
    # If True and nested_project_layout, copy matching label mask to outputs/overlays as *_gt_mask.png when present.
    export_gt_mask_when_labeled: bool = False

