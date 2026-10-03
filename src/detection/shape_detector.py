"""
Robust Geometric & Deep-Contrast Traffic Sign Detector.
Replaces naive color thresholding with:
  1. Skin & Body Exclusion (YCrCb color space filtering to reject human faces, arms, clothing)
  2. Edge Density & Internal Symbol Verification (rejects plain background, walls, solid shirts)
  3. Geometric Shape Analysis (Hough Circles + Circularity + Triangular Polygon Matching)
  4. Multi-Scale Localization (works on small road signs, phone screens, and printed paper)
"""

from typing import List, Tuple
import cv2
import numpy as np

from src.schema import DetectionResult, BoundingBox
from src.detection.base import BaseDetector


def is_skin_or_human_body(crop: np.ndarray) -> bool:
    """Detects if crop is predominantly human skin (face, neck, hands, body)."""
    if crop is None or crop.size == 0:
        return True
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    # Standard YCrCb skin tone ranges
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin_ratio = cv2.countNonZero(skin_mask) / float(crop.shape[0] * crop.shape[1])
    return skin_ratio > 0.35


def has_sign_symbol_contrast(crop: np.ndarray, min_edge_density: float = 0.035) -> bool:
    """Verifies that the crop contains an internal high-contrast symbol, not a plain solid color."""
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 140)
    density = cv2.countNonZero(edges) / float(crop.shape[0] * crop.shape[1])
    return density >= min_edge_density


class RobustTrafficSignDetector(BaseDetector):
    """
    High-precision traffic sign detector robust against human bodies, room lighting, and phone screen glare.
    """

    def __init__(self, min_size: int = 24, max_size: int = 500):
        self.min_size = min_size
        self.max_size = max_size

    def detect(self, image: np.ndarray, conf_threshold: float = 0.50) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        candidates: List[BoundingBox] = []

        # Strategy 1: Hough Circle Transform (for European circular speed limit, prohibitory, mandatory signs)
        # Parameterized for signs between 24px and 400px diameter
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, min(h, w) // 10),
            param1=100,
            param2=32, # Strictness of circle accumulator
            minRadius=self.min_size // 2,
            maxRadius=min(w, h, self.max_size) // 2
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                x1 = max(0, cx - r)
                y1 = max(0, cy - r)
                x2 = min(w, cx + r)
                y2 = min(h, cy + r)
                if (x2 - x1) >= self.min_size and (y2 - y1) >= self.min_size:
                    candidates.append(BoundingBox(x1, y1, x2, y2))

        # Strategy 2: High-contrast Edge Contours & Color-Invariant Shapes (for triangular & octagonal signs)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (self.min_size * self.min_size * 0.7) or area > (self.max_size * self.max_size):
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue

            circularity = 4 * np.pi * area / (peri * peri)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / float(bh)

            # Circular (circularity > 0.60) or Triangular / Octagonal signs (aspect ratio ~ 1.0)
            if 0.70 <= aspect_ratio <= 1.40:
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                num_v = len(approx)

                # Triangle (3 vertices), Octagon (8 vertices), or high circularity circle (> 0.65)
                if num_v in (3, 8) or circularity > 0.65:
                    candidates.append(BoundingBox(bx, by, bx + bw, by + bh))

        # Filter & Validate Candidates against Body/Background and Edge Density
        verified_detections: List[DetectionResult] = []
        seen_boxes = []

        for box in candidates:
            clamped = box.clamp(w, h)
            if clamped.width < self.min_size or clamped.height < self.min_size:
                continue

            # Extract crop
            crop = image[clamped.y1:clamped.y2, clamped.x1:clamped.x2]

            # 1. Reject if predominantly human skin / body
            if is_skin_or_human_body(crop):
                continue

            # 2. Reject if plain solid color / no internal contrast
            if not has_sign_symbol_contrast(crop, min_edge_density=0.035):
                continue

            # Check duplication with already verified boxes
            is_dup = False
            for sb in seen_boxes:
                # Compute simple IoU
                xA = max(clamped.x1, sb.x1)
                yA = max(clamped.y1, sb.y1)
                xB = min(clamped.x2, sb.x2)
                yB = min(clamped.y2, sb.y2)
                inter = max(0, xB - xA) * max(0, yB - yA)
                union = clamped.area + sb.area - inter
                if union > 0 and (inter / union) > 0.40:
                    is_dup = True
                    break

            if not is_dup:
                seen_boxes.append(clamped)
                verified_detections.append(DetectionResult(
                    bbox=clamped,
                    confidence=0.85,
                    detector_label="traffic_sign",
                    detector_class_id=0
                ))

        return verified_detections
