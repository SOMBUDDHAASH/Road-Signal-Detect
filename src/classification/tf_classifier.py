"""
TensorFlow / Keras Classifier Adapter for Traffic Sign Classification.
Maintained for compatibility with Member C's TensorFlow/Keras pipeline.
Preserves native OpenCV BGR color ordering and supports .keras and .h5 model formats.
"""

from typing import List, Optional, Tuple, Dict, Any
import os
import cv2
import numpy as np

from src.classification.base import BaseClassifier
from src.schema import ClassificationResult, SignCategory
from src.gtsrb_classes import (
    GTSRB_CLASSES,
    GTSRB_CATEGORIES,
    get_class_name,
    get_sign_category
)


class TensorFlowClassifier(BaseClassifier):
    """
    TensorFlow/Keras classifier adapter.
    Implements the standard BaseClassifier interface:
      classify(crop) -> ClassificationResult
      classify_batch(crops) -> List[ClassificationResult]

    Designed specifically for Member C's CNN architecture:
    - Input resolution: 32x32
    - Normalization: pixel / 255.0
    - Color Space: Native OpenCV BGR ordering (as trained by Member C)
    - Output: 43-class softmax probabilities
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        img_size: int = 32,
        color_mode: str = "bgr",  # "bgr" (Member C training) or "rgb"
        auto_fallback: bool = True,
        min_confidence: float = 0.30
    ):
        self.model_path = model_path or os.path.join("models", "traffic_sign_model.keras")
        if not os.path.exists(self.model_path):
            # Check secondary default path
            alt_path = os.path.join("weights", "classification", "traffic_sign_model.keras")
            if os.path.exists(alt_path):
                self.model_path = alt_path

        self.img_size = img_size
        self.color_mode = color_mode.lower()
        self.auto_fallback = auto_fallback
        self.min_confidence = min_confidence

        self._model = None
        self._has_tf = False
        self._fallback_classifier = None

        self._init_model()

    def _init_model(self):
        """Attempts to load the Keras / TensorFlow model with graceful fallback."""
        candidate_paths = [
            self.model_path,
            os.path.join("weights", "classification", "traffic_sign_model.keras"),
            os.path.join("classification", "models", "traffic_sign_model.keras"),
            os.path.join("models", "traffic_sign_model.keras"),
        ]
        resolved_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                resolved_path = p
                self.model_path = p
                break

        if resolved_path is None:
            if not self.auto_fallback:
                raise FileNotFoundError(f"Model file not found in paths: {candidate_paths}")
            self._init_fallback()
            return

        # 1. Try Keras 3 with torch backend (standard on Python 3.10-3.14)
        try:
            os.environ["KERAS_BACKEND"] = "torch"
            import keras
            self._model = keras.models.load_model(resolved_path)
            self._has_tf = True
            return
        except Exception:
            pass

        # 2. Try TensorFlow if installed
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(resolved_path)
            self._has_tf = True
            return
        except Exception:
            self._has_tf = False
            if not self.auto_fallback:
                raise
            self._init_fallback()

    def _init_fallback(self):
        """Initializes fallback PyTorch or heuristic classifier if TensorFlow is absent."""
        try:
            from src.classification.model import PyTorchClassifier
            self._fallback_classifier = PyTorchClassifier(auto_fallback=True, min_confidence=self.min_confidence)
        except Exception:
            from src.classification.mock import ColorHeuristicClassifier
            self._fallback_classifier = ColorHeuristicClassifier()

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses a single image crop:
        1. Resize to (img_size, img_size)
        2. Preserve BGR channel order (matches Member C's training) or convert to RGB
        3. Normalize pixel values to [0.0, 1.0]
        """
        if crop is None or crop.size == 0:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

        resized = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if self.color_mode == "rgb":
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def classify(self, crop: np.ndarray) -> ClassificationResult:
        """Classify a single traffic sign crop."""
        results = self.classify_batch([crop])
        return results[0]

    def classify_batch(self, crops: List[np.ndarray]) -> List[ClassificationResult]:
        """Classify a batch of traffic sign crops."""
        if not crops:
            return []

        # If running via fallback
        if self._model is None:
            if self._fallback_classifier is not None:
                return self._fallback_classifier.classify_batch(crops)
            return [
                ClassificationResult(
                    class_id=-1,
                    class_name="Unrecognized / Fallback",
                    confidence=0.0,
                    category=SignCategory.OTHER
                ) for _ in crops
            ]

        # Preprocess batch
        batch = np.array([self.preprocess(c) for c in crops], dtype=np.float32)

        try:
            probabilities = self._model.predict(batch, verbose=0)
        except Exception:
            if self._fallback_classifier:
                return self._fallback_classifier.classify_batch(crops)
            probabilities = np.zeros((len(crops), 43), dtype=np.float32)

        results: List[ClassificationResult] = []
        for probs in probabilities:
            pred_id = int(np.argmax(probs))
            conf = float(probs[pred_id])

            # Extract top 3 candidates
            top_k_indices = np.argsort(probs)[::-1][:3]
            top_k = [(int(idx), float(probs[idx])) for idx in top_k_indices]

            if conf < self.min_confidence:
                results.append(ClassificationResult(
                    class_id=-1,
                    class_name="Unrecognized / Background",
                    confidence=conf,
                    category=SignCategory.OTHER,
                    top_k=top_k
                ))
            else:
                cname = get_class_name(pred_id)
                cat = get_sign_category(pred_id)
                results.append(ClassificationResult(
                    class_id=pred_id,
                    class_name=cname,
                    confidence=conf,
                    category=cat,
                    top_k=top_k
                ))

        return results
