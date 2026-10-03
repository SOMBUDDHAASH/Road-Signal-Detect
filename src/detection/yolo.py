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
                from src.detection.shape_detector import RobustTrafficSignDetector
                self.fallback_detector = RobustTrafficSignDetector()
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError:
            if self.auto_fallback:
                from src.detection.shape_detector import RobustTrafficSignDetector
                self.fallback_detector = RobustTrafficSignDetector()
        except Exception as e:
            print(f"[Warning] Failed to load YOLO model from {self.model_path}: {e}")
            if self.auto_fallback:
                from src.detection.shape_detector import RobustTrafficSignDetector
                self.fallback_detector = RobustTrafficSignDetector()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        if not self.is_ready:
            if self.auto_fallback and self.fallback_detector is not None:
                if not self._warned:
                    print(f"[Info] YOLO weights not found at '{self.model_path}'. Automatically using RobustTrafficSignDetector fallback.")
                    self._warned = True
                return self.fallback_detector.detect(image, conf_threshold=conf_threshold)

            raise RuntimeError(
                f"YOLO detector weights not loaded from '{self.model_path}'. "
                "Ensure Member B has placed 'best.pt' in the weights directory or enable auto_fallback."
            )

        h, w = image.shape[:2]
        results = self.model.predict(source=image, conf=conf_threshold, verbose=False)

        disallowed_classes = {
            'person', 'human', 'face', 'body', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors'
        }

        detections: List[DetectionResult] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.model.names.get(cls_id, "traffic_sign").lower()

                # Guard 1: Reject person/body/everyday household classes
                if any(dis in cls_name for dis in disallowed_classes):
                    continue

                bbox = BoundingBox(
                    int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                ).clamp(w, h)

                # Guard 2: Skip degenerate or oversized bounding boxes
                if bbox.width < 12 or bbox.height < 12:
                    continue
                if bbox.width > (w * 0.70) or bbox.height > (h * 0.70):
                    continue

                # Guard 3: Aspect ratio check (road signs are roughly 1:1, triangles/circles/octagons)
                aspect_ratio = bbox.width / max(1, bbox.height)
                if aspect_ratio < 0.45 or aspect_ratio > 2.2:
                    continue

                detections.append(DetectionResult(
                    bbox=bbox,
                    confidence=conf,
                    detector_label=self.model.names.get(cls_id, "traffic_sign"),
                    detector_class_id=cls_id
                ))

        return detections
