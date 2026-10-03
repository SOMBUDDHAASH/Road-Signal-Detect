"""
Member C Module: GTSRB Traffic Sign Classifier (43 Classes).
Lead: Member C (Sneha Chakraborty) • Branch: feature/classification

Implements deep learning classification conforming to Member D's BaseClassifier contract.
Loads PyTorch / TorchScript weights, performs preprocessing, inference, and out-of-distribution rejection.
"""

from typing import List, Optional, Tuple, Dict, Any
import os
from pathlib import Path
import numpy as np
import cv2

# Ensure project root is reachable
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schema import ClassificationResult, SignCategory
from src.classification.base import BaseClassifier
from src.gtsrb_classes import get_class_name, get_sign_category


class StandaloneGTSRBClassifier(BaseClassifier):
    """
    Traffic Sign Classifier for GTSRB 43 classes.
    Loads TorchScript or standard PyTorch state dict and outputs ClassificationResults.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        img_size: int = 32,
        device: str = "cpu",
        min_confidence: float = 0.30
    ):
        # Default weight location in Module C or root weights
        module_c_weights = Path(__file__).resolve().parent / "weights" / "classifier.pt"
        root_weights = ROOT_DIR / "weights" / "classification" / "classifier.pt"

        if model_path:
            self.model_path = Path(model_path)
        elif module_c_weights.exists():
            self.model_path = module_c_weights
        elif root_weights.exists():
            self.model_path = root_weights
        else:
            self.model_path = module_c_weights

        self.img_size = img_size
        self.device = device
        self.min_confidence = min_confidence
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            print(f"[Warning] Classifier weights not found at: {self.model_path}")
            return

        try:
            import torch
            try:
                self.model = torch.jit.load(str(self.model_path), map_location=self.device)
                self.model.eval()
            except Exception:
                self.model = torch.load(str(self.model_path), map_location=self.device)
                if hasattr(self.model, "eval"):
                    self.model.eval()
            print(f"[Info] Classifier loaded successfully from: {self.model_path}")
        except Exception as e:
            print(f"[Error] Failed to load PyTorch model: {e}")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses a crop patch:
        - Resize to (img_size, img_size)
        - Convert BGR to RGB
        - Normalize to [0.0, 1.0]
        - Transpose to (1, 3, H, W)
        """
        resized = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def classify(self, crop: np.ndarray) -> ClassificationResult:
        if crop is None or crop.size == 0:
            return ClassificationResult(class_id=-1, class_name="Invalid Crop", confidence=0.0)

        if not self.is_ready:
            from src.classification.mock import ColorHeuristicClassifier
            fallback = ColorHeuristicClassifier()
            return fallback.classify(crop)

        import torch
        tensor_input = torch.from_numpy(self.preprocess(crop)).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1]
        best_id = int(top_indices[0])
        best_conf = float(probs[best_id])

        # Strict Out-of-Distribution / Non-sign rejection
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


if __name__ == "__main__":
    classifier = StandaloneGTSRBClassifier()
    print(f"Classifier ready: {classifier.is_ready}")

    # Test with dummy crop
    dummy_crop = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    res = classifier.classify(dummy_crop)
    print(f"Prediction: {res.class_id} -> {res.class_name} (conf={res.confidence:.2f})")
