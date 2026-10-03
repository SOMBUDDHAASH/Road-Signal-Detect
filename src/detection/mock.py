"""
Mock and Heuristic Detectors for testing and pre-model demonstration.
Allows the full pipeline to run immediately before Member B delivers YOLO weights.
"""

from typing import List, Tuple
import numpy as np
import cv2

from src.schema import DetectionResult, BoundingBox
from src.detection.base import BaseDetector


class MockDetector(BaseDetector):
    """
    Synthetic detector that generates fixed or parameterized bounding boxes.
    Ideal for unit tests and integration verification without deep learning dependencies.
    """

    def __init__(self, predefined_boxes: List[Tuple[int, int, int, int]] = None, default_conf: float = 0.92):
        self.predefined_boxes = predefined_boxes
        self.default_conf = default_conf

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        results = []

        if self.predefined_boxes:
            for (x1, y1, x2, y2) in self.predefined_boxes:
                bbox = BoundingBox(x1, y1, x2, y2).clamp(w, h)
                if self.default_conf >= conf_threshold and bbox.area > 0:
                    results.append(DetectionResult(
                        bbox=bbox,
                        confidence=self.default_conf,
                        detector_label="traffic_sign",
                        detector_class_id=0
                    ))
        else:
            # Default mock: center sign box covering ~20% of frame
            cx, cy = w // 2, h // 2
            size = min(w, h) // 4
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, cx + size // 2)
            y2 = min(h, cy + size // 2)
            bbox = BoundingBox(x1, y1, x2, y2)
            if self.default_conf >= conf_threshold and bbox.area > 0:
                results.append(DetectionResult(
                    bbox=bbox,
                    confidence=self.default_conf,
                    detector_label="traffic_sign",
                    detector_class_id=0
                ))

        return results


class ColorContourDetector(BaseDetector):
    """
    A functional Computer Vision detector based on HSV color segmentation and contour analysis.
    Road signs in Europe/GTSRB are predominantly:
      - Red circular/triangular (Prohibitory / Danger)
      - Blue circular/rectangular (Mandatory / Informational)
      - Yellow / Amber (Warning / Construction)
    """

    def __init__(self, min_area: int = 300, max_area: int = 150000):
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        # Dynamically scale min_area if image is very small (e.g. GTSRB test crops)
        effective_min_area = min(self.min_area, max(30, int(h * w * 0.04)))
        effective_max_area = max(self.max_area, int(h * w * 0.98))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red mask (wraps around 0/180 in HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        # Blue mask (Mandatory signs)
        lower_blue = np.array([95, 80, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Yellow mask (Danger / Road work)
        lower_yellow = np.array([15, 80, 70])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        combined_mask = cv2.bitwise_or(mask_red, mask_blue)
        combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)

        # Morphological opening and closing to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[DetectionResult] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < effective_min_area or area > effective_max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / bh

            # Traffic signs generally have an aspect ratio close to 1:1 (circular, triangular, square)
            if 0.60 <= aspect_ratio <= 1.65:
                # Approximate confidence based on contour density/solidity
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0.5
                confidence = float(min(0.98, max(0.55, solidity * 0.95)))

                if confidence >= conf_threshold:
                    bbox = BoundingBox(x, y, x + bw, y + bh).clamp(w, h)
                    detections.append(DetectionResult(
                        bbox=bbox,
                        confidence=confidence,
                        detector_label="traffic_sign",
                        detector_class_id=0
                    ))

        # If the input image is already a small cropped GTSRB sign (<150x150) and no contour was picked up,
        # detect the sign as the frame itself if it contains sign color pigment
        if not detections and max(h, w) <= 150:
            colored_pixels = cv2.countNonZero(combined_mask)
            if colored_pixels > (h * w * 0.10):
                margin_x = int(w * 0.05)
                margin_y = int(h * 0.05)
                bbox = BoundingBox(margin_x, margin_y, w - margin_x, h - margin_y)
                detections.append(DetectionResult(
                    bbox=bbox,
                    confidence=0.88,
                    detector_label="traffic_sign",
                    detector_class_id=0
                ))

        # Non-maximum suppression to remove overlapping boxes
        return self._apply_nms(detections, iou_threshold=0.35)

    def _apply_nms(self, detections: List[DetectionResult], iou_threshold: float) -> List[DetectionResult]:
        if not detections:
            return []

        boxes = [d.bbox.to_xyxy() for d in detections]
        scores = [d.confidence for d in detections]

        indices = cv2.dnn.NMSBoxes(
            bboxes=[[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes],
            scores=scores,
            score_threshold=0.0,
            nms_threshold=iou_threshold
        )

        if len(indices) == 0:
            return []

        kept_indices = indices.flatten() if hasattr(indices, 'flatten') else [i[0] for i in indices]
        return [detections[i] for i in kept_indices]
