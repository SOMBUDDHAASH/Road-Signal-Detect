"""
PyTorch & ONNX Classification Adapter for Member C (Classification Lead).
Supports loading trained PyTorch weights, TorchScript, or ONNX models trained on GTSRB.
"""

from typing import List, Optional
import os
import numpy as np
import cv2

from src.schema import ClassificationResult
from src.classification.base import BaseClassifier
from src.gtsrb_classes import get_class_name, get_sign_category


class PyTorchClassifier(BaseClassifier):
    """
    Adapter for Member C's trained CNN / MobileNetV3 / ResNet model on GTSRB (43 classes).
    Supports PyTorch (.pt/.pth) and TorchScript (.pt) model files.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        img_size: int = 32,
        device: str = "cpu",
        auto_fallback: bool = True,
        min_confidence: float = 0.65
    ):
        self.model_path = model_path or os.path.join("weights", "classification", "classifier.pt")
        self.img_size = img_size
        self.device = device
        self.auto_fallback = auto_fallback
        self.min_confidence = min_confidence
        self.fallback_classifier = None
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            if self.auto_fallback:
                from src.classification.mock import ColorHeuristicClassifier
                self.fallback_classifier = ColorHeuristicClassifier()
            return

        try:
            import torch
            try:
                self.model = torch.jit.load(self.model_path, map_location=self.device)
                self.model.eval()
            except Exception:
                self.model = torch.load(self.model_path, map_location=self.device)
                if hasattr(self.model, "eval"):
                    self.model.eval()
        except ImportError:
            if self.auto_fallback:
                from src.classification.mock import ColorHeuristicClassifier
                self.fallback_classifier = ColorHeuristicClassifier()
        except Exception as e:
            print(f"[Warning] Failed to load PyTorch classifier from {self.model_path}: {e}")
            if self.auto_fallback:
                from src.classification.mock import ColorHeuristicClassifier
                self.fallback_classifier = ColorHeuristicClassifier()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)

    def classify(self, crop: np.ndarray) -> ClassificationResult:
        if crop is None or crop.size == 0:
            return ClassificationResult(class_id=-1, class_name="Invalid Crop", confidence=0.0)

        if not self.is_ready:
            if self.auto_fallback and self.fallback_classifier is not None:
                return self.fallback_classifier.classify(crop)

            raise RuntimeError(
                f"Classification model weights not found at '{self.model_path}'. "
                "Ensure Member C places the model weights there or enable auto_fallback."
            )

        import torch
        tensor_input = torch.from_numpy(self.preprocess(crop)).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1]
        best_id = int(top_indices[0])
        best_conf = float(probs[best_id])

        # Strict Out-of-Distribution / Background rejection
        if best_conf < self.min_confidence:
            return ClassificationResult(
                class_id=-1,
                class_name="Unrecognized / Background",
                confidence=best_conf,
                category=SignCategory.OTHER
            )

        top_k = []
        for i in range(min(5, len(top_indices))):
            cid = int(top_indices[i])
            top_k.append((cid, get_class_name(cid), float(probs[cid])))

        return ClassificationResult(
            class_id=best_id,
            class_name=get_class_name(best_id),
            confidence=best_conf,
            category=get_sign_category(best_id),
            top_k=top_k
        )
