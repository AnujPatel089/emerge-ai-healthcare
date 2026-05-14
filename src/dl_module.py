"""
Deep Learning Medical Image Module
- EfficientNet-B0 backbone (transfer learning friendly)
- Multi-head: wound type classification + infection severity regression
- Plays alongside cv_module.py (cv_module = classical features, dl_module = neural)
"""

import os
import io
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

logger = logging.getLogger(__name__)

# ---------- Class taxonomies ----------
WOUND_CLASSES = [
    "abrasion",
    "laceration",
    "burn",
    "surgical_incision",
    "ulcer",
    "puncture",
    "contusion",
    "healthy_skin",
]

SEVERITY_LEVELS = ["none", "mild", "moderate", "severe", "critical"]


# ---------- Model definition ----------
class MedicalImageNet(nn.Module):
    """
    EfficientNet-B0 backbone with two heads:
      - wound_head: classification over WOUND_CLASSES
      - infection_head: 5-class infection severity (ordinal-friendly)
    """

    def __init__(self, num_wound_classes: int = len(WOUND_CLASSES),
                 num_severity_classes: int = len(SEVERITY_LEVELS),
                 pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Strip the classifier so we keep the 1280-d feature vector.
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.dropout = nn.Dropout(p=0.3)
        self.wound_head = nn.Linear(in_features, num_wound_classes)
        self.infection_head = nn.Linear(in_features, num_severity_classes)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        feats = self.dropout(feats)
        return {
            "features": feats,
            "wound_logits": self.wound_head(feats),
            "infection_logits": self.infection_head(feats),
        }


# ---------- Inference wrapper ----------
class DLImageAnalyzer:
    """
    Loads MedicalImageNet and exposes a single .analyze(image_bytes) call.
    Falls back to ImageNet-pretrained weights if no fine-tuned checkpoint
    is present — keeps the pipeline runnable end-to-end during development.
    """

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = MedicalImageNet(pretrained=True).to(self.device)

        self.weights_path = weights_path or os.getenv(
            "EMERGEAI_DL_WEIGHTS", "models/medical_imagenet.pt"
        )
        self._load_weights()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_weights(self):
        if os.path.exists(self.weights_path):
            try:
                state = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                logger.info("Loaded fine-tuned weights from %s", self.weights_path)
            except Exception as e:
                logger.warning("Failed to load weights (%s); using ImageNet init.", e)
        else:
            logger.warning(
                "No checkpoint at %s — running with ImageNet-pretrained backbone. "
                "Predictions are illustrative until a clinical dataset is fine-tuned in.",
                self.weights_path,
            )

    @torch.no_grad()
    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        out = self.model(tensor)
        wound_probs = F.softmax(out["wound_logits"], dim=1)[0].cpu().numpy()
        infection_probs = F.softmax(out["infection_logits"], dim=1)[0].cpu().numpy()

        wound_idx = int(np.argmax(wound_probs))
        infection_idx = int(np.argmax(infection_probs))

        # Ordinal severity score in [0, 1] — useful for downstream triage fusion.
        severity_weights = np.linspace(0, 1, len(SEVERITY_LEVELS))
        severity_score = float(np.dot(infection_probs, severity_weights))

        return {
            "wound_class": WOUND_CLASSES[wound_idx],
            "wound_confidence": float(wound_probs[wound_idx]),
            "wound_distribution": {
                cls: float(p) for cls, p in zip(WOUND_CLASSES, wound_probs)
            },
            "infection_severity": SEVERITY_LEVELS[infection_idx],
            "infection_confidence": float(infection_probs[infection_idx]),
            "infection_distribution": {
                lvl: float(p) for lvl, p in zip(SEVERITY_LEVELS, infection_probs)
            },
            "severity_score": severity_score,  # continuous 0–1
            "feature_vector": out["features"][0].cpu().numpy().tolist(),
            "model": "EfficientNet-B0 (multi-head)",
            "device": str(self.device),
        }


# Singleton accessor — load once per process.
_analyzer: Optional[DLImageAnalyzer] = None


def get_dl_analyzer() -> DLImageAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = DLImageAnalyzer()
    return _analyzer
