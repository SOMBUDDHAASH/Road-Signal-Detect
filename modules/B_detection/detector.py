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

        disallowed_classes = {
            'person', 'human', 'face', 'body', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'cell phone'
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

                if any(dis in cls_name for dis in disallowed_classes):
                    continue

                bbox = BoundingBox(
                    int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                ).clamp(w, h)

                # Skip degenerate or oversized bounding boxes
                if bbox.width < 12 or bbox.height < 12:
                    continue
                if bbox.width > (w * 0.70) or bbox.height > (h * 0.70):
                    continue

                aspect_ratio = bbox.width / max(1, bbox.height)
                if aspect_ratio < 0.45 or aspect_ratio > 2.2:
                    continue

                detections.append(DetectionResult(
                    bbox=bbox,
                    confidence=conf,
                    detector_label=self.model.names.get(cls_id, "traffic_sign"),
                    detector_class_id=cls_id
                ))

        # Stage 2: High-Recall Color & Shape Region Proposals with Robust Semantic Guards
        # Captures genuine traffic signs (danger triangles, blue circles, red circles, yellow diamonds)
        # while strictly rejecting human faces/bodies, room walls, and ambient background noise
        import cv2
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, np.array([0, 65, 45]), np.array([10, 255, 255])) | cv2.inRange(hsv, np.array([170, 65, 45]), np.array([180, 255, 255]))
        blue_mask = cv2.inRange(hsv, np.array([100, 65, 45]), np.array([130, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 65, 45]), np.array([35, 255, 255]))
        combined = red_mask | blue_mask | yellow_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = float(w * h)

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Guard 1: Size limits (compact road signs, never huge room/body-sized bounding boxes)
            if bw < 24 or bh < 24:
                continue
            if bw > (w * 0.45) or bh > (h * 0.50):
                continue
            box_area = float(bw * bh)
            if box_area > (frame_area * 0.18):
                continue

            # Guard 2: Aspect ratio consistency (circles, octagons, triangles, diamonds, standard rectangles)
            aspect_ratio = bw / float(bh)
            if aspect_ratio < 0.52 or aspect_ratio > 1.88:
                continue

            # Guard 3: Geometric Solidity & Extent (rejects wispy, hollow human/room lighting contours)
            cnt_area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = cnt_area / max(1.0, hull_area)
            extent = cnt_area / max(1.0, box_area)
            if solidity < 0.65 or extent < 0.28:
                continue

            crop = image[by:by+bh, bx:bx+bw]
            if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
                continue

            # Guard 4: Skin color exclusion (human face/body rejection)
            ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
            skin_ratio = cv2.countNonZero(skin_mask) / float(crop.shape[0] * crop.shape[1])
            if skin_ratio > 0.15:
                continue

            # Guard 5: Texture & Edge energy (rejects flat walls, ceilings, single-color clothing)
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
            if lap_var < 15.0:
                continue

            # Guard 6: Road Sign Color Signature Concentration
            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            c_red = cv2.inRange(crop_hsv, np.array([0, 65, 45]), np.array([10, 255, 255])) | cv2.inRange(crop_hsv, np.array([170, 65, 45]), np.array([180, 255, 255]))
            c_blue = cv2.inRange(crop_hsv, np.array([100, 65, 45]), np.array([130, 255, 255]))
            c_yellow = cv2.inRange(crop_hsv, np.array([15, 65, 45]), np.array([35, 255, 255]))
            sign_color_ratio = cv2.countNonZero(c_red | c_blue | c_yellow) / float(crop.shape[0] * crop.shape[1])
            if sign_color_ratio < 0.16:
                continue

            candidate_box = BoundingBox(bx, by, bx + bw, by + bh).clamp(w, h)
            overlap = False
            for existing in detections:
                if existing.bbox.iou(candidate_box) > 0.35:
                    overlap = True
                    break
            if not overlap:
                detections.append(DetectionResult(
                    bbox=candidate_box,
                    confidence=0.75,
                    detector_label="traffic_sign",
                    detector_class_id=0
                ))

        # Stage 3: Global Non-Maximum Suppression (NMS) to eliminate duplicate boxes
        detections.sort(key=lambda d: d.confidence, reverse=True)
        final_detections: List[DetectionResult] = []
        for det in detections:
            if not any(det.bbox.iou(kept.bbox) > 0.40 for kept in final_detections):
                final_detections.append(det)

        return final_detections



if __name__ == "__main__":
    import cv2
    detector = StandaloneYOLODetector()
    print(f"Detector ready: {detector.is_ready}")

    # Test with dummy image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(dummy_frame, (320, 240), 50, (0, 0, 255), -1)
    results = detector.detect(dummy_frame, conf_threshold=0.2)
    print(f"Detections found: {len(results)}")
