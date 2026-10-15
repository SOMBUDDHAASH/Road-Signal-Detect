"""YOLOv8 Detection Engine (Member B)
Provides high-speed real-time traffic sign localization using fine-tuned YOLOv8.
"""

import os
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    YOLO_MODEL_PATH,
    DEFAULT_DET_CONF,
    DEFAULT_DET_IOU,
    CLASS_NAMES,
    PROJECT_ROOT,
)
from src.schema import DetectionResult, BoundingBox


class YOLODetector:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the YOLOv8 detector with model weights."""
        resolved_path = self._resolve_model_path(model_path)
        print(f"[YOLO] Loading detector weights from: {resolved_path}")
        self.model = YOLO(str(resolved_path))
        self.model_path = resolved_path
        # Extract class names mapping from model or fallback to config
        self.names = getattr(self.model, "names", CLASS_NAMES)

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        """Find the best available YOLO model file."""
        if model_path and os.path.exists(model_path):
            return Path(model_path)

        if YOLO_MODEL_PATH.exists():
            return YOLO_MODEL_PATH

        # Fallbacks: check project root or download pretrained yolov8n
        fallback_root = PROJECT_ROOT / "yolov8n.pt"
        if fallback_root.exists():
            return fallback_root

        return Path("yolov8n.pt")

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = DEFAULT_DET_CONF,
        iou_threshold: float = DEFAULT_DET_IOU,
    ) -> List[DetectionResult]:
        """Run YOLO inference on a single BGR image.

        Returns:
            List[DetectionResult]: list of detected signs with bounding box & confidence.
        """
        if image is None or image.size == 0:
            return []

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        h, w = image.shape[:2]

        for box in first_res.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            # Map class name
            if isinstance(self.names, dict):
                cls_name = self.names.get(cls_id, CLASS_NAMES.get(cls_id, f"Sign_{cls_id}"))
            elif isinstance(self.names, list) and cls_id < len(self.names):
                cls_name = self.names[cls_id]
            else:
                cls_name = CLASS_NAMES.get(cls_id, f"Sign_{cls_id}")

            bbox = BoundingBox(
                x1=int(xyxy[0]),
                y1=int(xyxy[1]),
                x2=int(xyxy[2]),
                y2=int(xyxy[3]),
            ).clamp(w, h)

            detections.append(
                DetectionResult(
                    bbox=bbox,
                    confidence=round(conf, 4),
                    class_id=cls_id,
                    class_name=cls_name,
                    detector_source="YOLOv8",
                )
            )

        return detections

    def crop_detections(
        self, image: np.ndarray, detections: List[DetectionResult], padding: int = 4
    ) -> List[np.ndarray]:
        """Crop all detected sign regions from the input image."""
        crops = []
        for det in detections:
            crop = det.bbox.crop(image, padding=padding)
            if crop is not None:
                crops.append(crop)
        return crops

    def draw_detections(
        self, image: np.ndarray, detections: List[DetectionResult]
    ) -> np.ndarray:
        """Draw bounding boxes and labels onto an image copy."""
        canvas = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            label = f"{det.class_name} {det.confidence:.2f}"

            # Box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label banner
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_banner = max(y1, th + 5)
            cv2.rectangle(canvas, (x1, y_banner - th - 5), (x1 + tw + 4, y_banner + bl), (0, 255, 0), cv2.FILLED)
            cv2.putText(canvas, label, (x1 + 2, y_banner), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas


# Convenience functional interface matching Member B's original API
def load_model(model_path=None):
    return YOLODetector(model_path)

def detect_signs(detector, image, conf=DEFAULT_DET_CONF, iou=DEFAULT_DET_IOU):
    if isinstance(detector, YOLODetector):
        return detector.detect(image, conf, iou)
    det = YOLODetector()
    return det.detect(image, conf, iou)
