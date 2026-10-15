"""PyTorch CNN Classifier Engine.
Provides native PyTorch / TorchScript classification as a companion or fallback.
"""

import os
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    PYTORCH_MODEL_PATH,
    CLASSIFICATION_IMG_SIZE,
    CLASS_NAMES,
    SPEED_LIMIT_MAP,
    get_category_for_class,
)
from src.schema import ClassificationResult, SignCategory


class PyTorchSignClassifier:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize PyTorch TorchScript model."""
        resolved = Path(model_path) if model_path else PYTORCH_MODEL_PATH
        if not resolved.exists():
            raise FileNotFoundError(f"PyTorch model not found at {resolved}")

        print(f"[CLASSIFIER] Loading PyTorch model from: {resolved}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(resolved), map_location=self.device)
        self.model.eval()
        self.img_size = CLASSIFICATION_IMG_SIZE

    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Preprocess BGR/RGB crop to PyTorch NCHW tensor (1, 3, 32, 32) float32 [0, 1]."""
        if crop is None or crop.size == 0:
            raise ValueError("Input crop is empty or None")

        if len(crop.shape) == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0

        # HWC -> CHW -> NCHW
        tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def classify(self, crop: np.ndarray, top_k: int = 3) -> ClassificationResult:
        """Classify a single traffic sign crop."""
        tensor = self.preprocess_crop(crop)
        with torch.no_grad():
            output = self.model(tensor)
            # Hierarchical models output tuple (head_fine, head_category)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        category_str = get_category_for_class(class_id)
        category = getattr(SignCategory, category_str, SignCategory.OTHER)
        speed_limit = SPEED_LIMIT_MAP.get(class_id)

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_list = [
            (int(idx), CLASS_NAMES.get(int(idx), f"Class {idx}"), float(probs[idx]))
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
