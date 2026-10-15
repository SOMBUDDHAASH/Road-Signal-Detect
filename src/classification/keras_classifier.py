"""Keras 3 CNN Classifier Engine (Member C)
Loads and serves Member C's trained GTSRB CNN model (traffic_sign_model.keras).
Compatible with PyTorch backend or TensorFlow backend.
Achieves 99.34% internal test accuracy and 95.20% official GTSRB benchmark accuracy.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

# Ensure PyTorch backend is set for Keras 3
os.environ["KERAS_BACKEND"] = "torch"
import keras

from src.config import (
    KERAS_MODEL_PATH,
    CLASSIFICATION_IMG_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    SPEED_LIMIT_MAP,
    get_category_for_class,
)
from src.schema import ClassificationResult, SignCategory


class KerasSignClassifier:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize Keras classifier with pre-trained .keras weights."""
        resolved = Path(model_path) if model_path else KERAS_MODEL_PATH
        if not resolved.exists():
            raise FileNotFoundError(
                f"Keras model not found at {resolved}. "
                "Ensure traffic_sign_model.keras is in models/classification/."
            )

        print(f"[CLASSIFIER] Loading Keras CNN model from: {resolved}")
        self.model = keras.models.load_model(str(resolved))
        self.img_size = CLASSIFICATION_IMG_SIZE

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess a cropped image to (1, 32, 32, 3) float32 [0, 1] BGR format matching Member C CNN."""
        if crop is None or crop.size == 0:
            raise ValueError("Input crop is empty or None")

        # Member C's model was trained on OpenCV BGR format
        # If grayscale, convert to BGR
        if len(crop.shape) == 2:
            bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            bgr = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        else:
            bgr = crop

        # Resize to 32x32
        resized = cv2.resize(bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        norm = resized.astype(np.float32) / 255.0

        # Expand batch dimension
        return np.expand_dims(norm, axis=0)

    def classify(self, crop: np.ndarray, top_k: int = 3) -> ClassificationResult:
        """Classify a single traffic sign crop.

        Returns:
            ClassificationResult containing class_id, class_name, confidence, category, top_k.
        """
        tensor = self.preprocess_crop(crop)
        preds = self.model.predict(tensor, verbose=0)[0]

        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        category_str = get_category_for_class(class_id)
        category = getattr(SignCategory, category_str, SignCategory.OTHER)
        speed_limit = SPEED_LIMIT_MAP.get(class_id)

        # Top-k predictions
        top_indices = np.argsort(preds)[::-1][:top_k]
        top_k_list = [
            (int(idx), CLASS_NAMES.get(int(idx), f"Class {idx}"), float(preds[idx]))
            for idx in top_indices
        ]

        return ClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=round(confidence, 4),
            category=category,
            top_k=top_k_list,
            speed_limit=speed_limit,
        )

    def classify_batch(self, crops: List[np.ndarray]) -> List[ClassificationResult]:
        """Classify a batch of cropped sign images."""
        if not crops:
            return []

        processed = []
        valid_indices = []
        for i, c in enumerate(crops):
            try:
                processed.append(self.preprocess_crop(c)[0])
                valid_indices.append(i)
            except Exception:
                continue

        if not processed:
            return []

        batch_tensor = np.array(processed, dtype=np.float32)
        all_preds = self.model.predict(batch_tensor, verbose=0)

        results = []
        for preds in all_preds:
            cid = int(np.argmax(preds))
            conf = float(preds[cid])
            name = CLASS_NAMES.get(cid, f"Class {cid}")
            cat_str = get_category_for_class(cid)
            cat = getattr(SignCategory, cat_str, SignCategory.OTHER)
            speed_limit = SPEED_LIMIT_MAP.get(cid)

            top_indices = np.argsort(preds)[::-1][:3]
            top_k = [(int(idx), CLASS_NAMES.get(int(idx), str(idx)), float(preds[idx])) for idx in top_indices]

            results.append(
                ClassificationResult(
                    class_id=cid,
                    class_name=name,
                    confidence=round(conf, 4),
                    category=cat,
                    top_k=top_k,
                    speed_limit=speed_limit,
                )
            )

        return results
