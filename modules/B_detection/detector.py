"""
Member B Module: Real-Time Traffic Sign Detection (YOLOv8).
Lead: Member B (Shaurya Vaid) • Branch: feature/detection

Implements YOLOv8 detection conforming to Member D's BaseDetector contract.
Can be run standalone or directly plugged into the integration pipeline.
"""

from typing import List, Optional, Tuple, Dict, Any
import os
from pathlib import Path
import numpy as np

# Ensure project root is reachable
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schema import DetectionResult, BoundingBox
from src.detection.base import BaseDetector


class StandaloneYOLODetector(BaseDetector):
    """
    YOLOv8 Detection wrapper for traffic sign localization.
    Loads Ultralytics YOLO model (.pt or .onnx) and outputs bounding boxes.
    """

    def __init__(self, model_path: Optional[str] = None):
        # Default weight location in Module B or root weights
        module_b_weights = Path(__file__).resolve().parent / "weights" / "best.pt"
        root_weights = ROOT_DIR / "weights" / "detection" / "best.pt"

        if model_path:
            self.model_path = Path(model_path)
        elif module_b_weights.exists():
            self.model_path = module_b_weights
        elif root_weights.exists():
            self.model_path = root_weights
        else:
            self.model_path = module_b_weights

        self.model = None
        self._load_weights()

    def _load_weights(self):
        if not self.model_path.exists():
            print(f"[Warning] YOLO weights not found at: {self.model_path}")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            print(f"[Info] YOLO model loaded successfully from: {self.model_path}")
        except Exception as e:
            print(f"[Error] Failed to initialize YOLO model: {e}")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def detect(self, image: np.ndarray, conf_threshold: float = 0.4) -> List[DetectionResult]:
        """
        Executes YOLO inference on frame, filters by confidence, and returns DetectionResults.
        """
        if image is None or image.size == 0:
            return []

        if not self.is_ready:
            # Fallback to shape detector if YOLO model is not ready
            from src.detection.shape_detector import RobustTrafficSignDetector
            fallback = RobustTrafficSignDetector()
            return fallback.detect(image, conf_threshold=conf_threshold)

        h, w = image.shape[:2]
        results = self.model.predict(source=image, conf=conf_threshold, verbose=False)

        detections: List[DetectionResult] = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                bbox = BoundingBox(
                    int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                ).clamp(w, h)

                # Skip degenerate bounding boxes
                if bbox.width < 10 or bbox.height < 10:
                    continue

                detections.append(DetectionResult(
                    bbox=bbox,
                    confidence=conf,
                    detector_label="traffic_sign",
                    detector_class_id=cls_id
                ))

        return detections


if __name__ == "__main__":
    import cv2
    detector = StandaloneYOLODetector()
    print(f"Detector ready: {detector.is_ready}")

    # Test with dummy image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(dummy_frame, (320, 240), 50, (0, 0, 255), -1)
    results = detector.detect(dummy_frame, conf_threshold=0.2)
    print(f"Detections found: {len(results)}")
