from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

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


class SegmentationInference:
    """
    DL-assisted inference for lead/crack masks.
    - If checkpoint exists: expects output tensor [B, 2, H, W].
    - Otherwise: uses torchvision Deeplab backbone as a generic feature prior
      and derives lead/crack masks through thresholded responses.
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

        self.model = self._build_model(checkpoint_path, use_torch_backbone)
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _build_model(self, checkpoint_path: Optional[str], use_torch_backbone: bool):
        if deeplabv3_resnet50 is None or torch is None:
            return None

        weights = DeepLabV3_ResNet50_Weights.DEFAULT if use_torch_backbone else None
        model = deeplabv3_resnet50(weights=weights)
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)

        return model

    def predict(self, image_bgr: np.ndarray) -> InferenceOutput:
        if self.model is None or torch is None or self.transform is None:
            return self._cv_fallback(image_bgr)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)["out"]
            prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        lead_prob = prob[0]
        crack_prob = prob[1]

        lead_mask = (lead_prob >= self.confidence_threshold).astype(np.uint8) * 255
        crack_mask = (crack_prob >= self.confidence_threshold).astype(np.uint8) * 255

        return InferenceOutput(
            lead_mask=lead_mask,
            crack_mask=crack_mask,
            lead_score=float(np.mean(lead_prob)),
            crack_score=float(np.mean(crack_prob)),
        )

    def _cv_fallback(self, image_bgr: np.ndarray) -> InferenceOutput:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Lead prior: bright elongated structures
        lead_mask = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -4
        )
        lead_mask = cv2.morphologyEx(
            lead_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )

        # Crack prior: dark thin discontinuities highlighted by blackhat + edges
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
        return {"engine": "torch-deeplab", "device": self.device}

