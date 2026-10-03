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

