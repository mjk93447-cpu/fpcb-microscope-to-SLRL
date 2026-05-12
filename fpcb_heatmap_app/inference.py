from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        deeplabv3_resnet50,
    )
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    T = None
    DeepLabV3_ResNet50_Weights = None
    deeplabv3_resnet50 = None


@dataclass
class InferenceOutput:
    lead_mask: np.ndarray
    crack_mask: np.ndarray
    lead_score: float
    crack_score: float


def _state_dict_from_checkpoint(ck: Any) -> dict[str, Any]:
    if isinstance(ck, dict) and "state_dict" in ck:
        return ck["state_dict"]
    if isinstance(ck, dict):
        return ck
    return {}


def _infer_num_classes_from_state(sd: dict[str, Any], ck_meta: dict[str, Any]) -> int:
    nc = ck_meta.get("num_classes")
    if nc in (2, 3):
        return int(nc)
    w = None
    for k, v in sd.items():
        if k.endswith("classifier.4.weight") and hasattr(v, "shape"):
            w = v
            break
    if w is not None and len(w.shape) >= 1:
        return int(w.shape[0])
    return 2


class SegmentationInference:
    """
    DL-assisted inference for lead/crack masks.
    - 3-class checkpoint (recommended): logits order background / lead / crack.
    - 2-class legacy: background / crack; lead map falls back to OpenCV heuristics.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        confidence_threshold: float = 0.45,
        use_gpu: bool = True,
        use_torch_backbone: bool = False,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.device = "cpu"
        self.model = None
        self.transform = None
        self._num_classes = 2

        if torch is None:
            return
        if not checkpoint_path and not use_torch_backbone:
            return

        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"

        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
            ]
        )

        self.model, self._num_classes = self._build_model(checkpoint_path, use_torch_backbone)
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _build_model(self, checkpoint_path: Optional[str], use_torch_backbone: bool):
        if deeplabv3_resnet50 is None or torch is None:
            return None, 2

        weights = DeepLabV3_ResNet50_Weights.DEFAULT if use_torch_backbone else None
        model = deeplabv3_resnet50(weights=weights)

        num_classes = 2
        ck_meta: dict[str, Any] = {}
        if checkpoint_path:
            p = Path(checkpoint_path)
            if p.is_file():
                state_obj = torch.load(str(p), map_location="cpu")
                sd = _state_dict_from_checkpoint(state_obj)
                if isinstance(state_obj, dict):
                    ck_meta = state_obj.get("meta") or {}
                    if not isinstance(ck_meta, dict):
                        ck_meta = {}
                num_classes = _infer_num_classes_from_state(sd, ck_meta)
                model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
                model.load_state_dict(sd, strict=False)
            else:
                model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
        else:
            model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)

        return model, num_classes

    def predict(self, image_bgr: np.ndarray) -> InferenceOutput:
        if self.model is None or torch is None or self.transform is None:
            return self._cv_fallback(image_bgr)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)["out"]
            prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        if self._num_classes >= 3:
            lead_prob = prob[1]
            crack_prob = prob[2]
            lead_mask = (lead_prob >= self.confidence_threshold).astype(np.uint8) * 255
            crack_mask = (crack_prob >= self.confidence_threshold).astype(np.uint8) * 255
            return InferenceOutput(
                lead_mask=lead_mask,
                crack_mask=crack_mask,
                lead_score=float(np.mean(lead_prob)),
                crack_score=float(np.mean(crack_prob)),
            )

        crack_prob = prob[1]
        crack_mask = (crack_prob >= self.confidence_threshold).astype(np.uint8) * 255
        fb = self._cv_fallback(image_bgr)
        return InferenceOutput(
            lead_mask=fb.lead_mask,
            crack_mask=crack_mask,
            lead_score=fb.lead_score,
            crack_score=float(np.mean(crack_prob)),
        )

    def _cv_fallback(self, image_bgr: np.ndarray) -> InferenceOutput:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        lead_mask = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -4
        )
        lead_mask = cv2.morphologyEx(
            lead_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )

        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, np.ones((7, 7), np.uint8))
        edges = cv2.Canny(blackhat, 30, 120)
        crack_mask = cv2.morphologyEx(
            edges, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8), iterations=1
        )

        return InferenceOutput(
            lead_mask=lead_mask,
            crack_mask=crack_mask,
            lead_score=float(np.mean(lead_mask > 0)),
            crack_score=float(np.mean(crack_mask > 0)),
        )

    def info(self) -> Dict[str, str]:
        if self.model is None:
            return {"engine": "opencv-fallback", "device": "cpu"}
        return {"engine": "torch-deeplab", "device": self.device, "num_classes": str(self._num_classes)}
