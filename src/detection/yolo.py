"""
YOLO Detection Adapter for Member B (Detection Lead).
Supports loading Ultralytics YOLOv8/v11 models or TorchScript/ONNX models.
"""

from typing import List, Optional
import os
import numpy as np

from src.schema import DetectionResult, BoundingBox
from src.detection.base import BaseDetector


class YOLODetector(BaseDetector):
    """
    Adapter for Member B's YOLO detector.
    Loads YOLO model weights (.pt or .onnx) and executes inference.
    If weights are not yet present, automatically falls back to ColorContourDetector.
    """

    def __init__(self, model_path: Optional[str] = None, auto_fallback: bool = True):
        self.model_path = model_path or os.path.join("weights", "detection", "best.pt")
        self.auto_fallback = auto_fallback
        self.model = None
        self.fallback_detector = None
        self._warned = False
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            if self.auto_fallback:
                from src.detection.mock import ColorContourDetector
                self.fallback_detector = ColorContourDetector()
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError:
            if self.auto_fallback:
                from src.detection.mock import ColorContourDetector
                self.fallback_detector = ColorContourDetector()
        except Exception as e:
            print(f"[Warning] Failed to load YOLO model from {self.model_path}: {e}")
            if self.auto_fallback:
                from src.detection.mock import ColorContourDetector
                self.fallback_detector = ColorContourDetector()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        if not self.is_ready:
            if self.auto_fallback and self.fallback_detector is not None:
                if not self._warned:
                    print(f"[Info] YOLO weights not found at '{self.model_path}'. Automatically using ColorContourDetector fallback.")
                    self._warned = True
                return self.fallback_detector.detect(image, conf_threshold=conf_threshold)

            raise RuntimeError(
                f"YOLO detector weights not loaded from '{self.model_path}'. "
                "Ensure Member B has placed 'best.pt' in the weights directory or enable auto_fallback."
            )

        h, w = image.shape[:2]
        results = self.model.predict(source=image, conf=conf_threshold, verbose=False)

        detections: List[DetectionResult] = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                bbox = BoundingBox(
                    int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                ).clamp(w, h)

                detections.append(DetectionResult(
                    bbox=bbox,
                    confidence=conf,
                    detector_label="traffic_sign",
                    detector_class_id=cls_id
                ))

        return detections
