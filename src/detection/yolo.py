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
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join("weights", "detection", "best.pt")
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            # Model file not found yet; will raise or fallback gracefully
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError:
            pass
        except Exception as e:
            print(f"[Warning] Failed to load YOLO model from {self.model_path}: {e}")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        if not self.is_ready:
            raise RuntimeError(
                f"YOLO detector weights not loaded from '{self.model_path}'. "
                "Ensure Member B has placed 'best.pt' in the weights directory or use ColorContourDetector."
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
