"""
Plague Model Secondary Traffic Sign Detector.
Implements seed-based color-pair adjacency discovery, cellular infection spreading,
unexpected-color cancellation (immune response), shape validation,
and OCR number/text extraction.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2

from src.schema import DetectionResult, BoundingBox, SignCategory, ClassificationResult
from src.detection.color_taxonomy import (
    ROAD_SIGN_COLOR_COMBINATIONS,
    COLOR_HSV_RANGES,
    get_color_mask,
    get_skin_mask,
    is_skin_dominated,
    ColorCombinationRule
)
from src.detection.ocr_engine import RoadSignOCREngine, SignTextResult
from src.dataset.common_signs_100 import COMMON_ROAD_SIGNS_100, RoadSignInfo


class PlagueSecondaryDetector:
    """
    Bio-inspired 'Plague' secondary localization and recognition engine.
    1. Finds seed infection sites where valid sign color pairs touch.
    2. Spreads infection outward across the valid palette.
    3. Cancels candidates if unexpected/forbidden colors (skin, foliage, non-sign textures) are encountered.
    4. Runs OCR number & string detection on surviving regions.
    """

    def __init__(
        self,
        min_sign_size: int = 18,
        max_sign_ratio: float = 0.70,
        enable_ocr: bool = True
    ):
        self.min_sign_size = min_sign_size
        self.max_sign_ratio = max_sign_ratio
        self.enable_ocr = enable_ocr
        self.ocr_engine = RoadSignOCREngine() if enable_ocr else None

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.35
    ) -> List[Tuple[DetectionResult, ClassificationResult]]:
        """
        Execute the plague spreading model across all registered color pairs.
        Returns a list of (DetectionResult, ClassificationResult) pairs.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skin_mask = get_skin_mask(image)

        candidates: List[Tuple[BoundingBox, float, str, int, SignCategory, str]] = []
        # Structure: (bbox, conf, class_name, class_id, category, detection_source)

        kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Test each major color pair rule
        for rule in ROAD_SIGN_COLOR_COMBINATIONS:
            mask_pri = get_color_mask(hsv, rule.primary_color)
            mask_sec = get_color_mask(hsv, rule.secondary_color)

            if cv2.countNonZero(mask_pri) < 20 or cv2.countNonZero(mask_sec) < 20:
                continue

            # Stage 1: Seed Discovery
            # Identify border boundary where primary color touches secondary color
            dilated_pri = cv2.dilate(mask_pri, kernel_3)
            seed_boundary = cv2.bitwise_and(dilated_pri, mask_sec)

            if cv2.countNonZero(seed_boundary) < 8:
                continue

            # Stage 2: Infection / Plague Spreading
            # Spread infection across the combined palette (primary + secondary)
            combined_palette = cv2.bitwise_or(mask_pri, mask_sec)
            closed_palette = cv2.morphologyEx(combined_palette, cv2.MORPH_CLOSE, kernel_5)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_palette)

            for i in range(1, num_labels):
                bx, by, bw, bh, area = stats[i]

                # Size constraint
                if bw < self.min_sign_size or bh < self.min_sign_size:
                    continue
                if bw > (w * self.max_sign_ratio) or bh > (h * self.max_sign_ratio):
                    continue

                # Aspect ratio constraint (traffic signs are circles, squares, triangles, octagons)
                aspect = bw / float(bh)
                if aspect < 0.48 or aspect > 2.1:
                    continue

                # Verify infection seed: component must contain active seed boundary pixels
                comp_mask = (labels == i).astype(np.uint8) * 255
                has_seed = cv2.countNonZero(cv2.bitwise_and(comp_mask, seed_boundary)) > 4
                if not has_seed:
                    continue

                # Stage 3: The Immune Response / Unexpected Color Cancellation
                crop_bgr = image[by:by+bh, bx:bx+bw]
                crop_skin = skin_mask[by:by+bh, bx:bx+bw]

                # Cancellation Guard A: Skin / Body
                skin_ratio = cv2.countNonZero(crop_skin) / float(bw * bh)
                if skin_ratio > 0.18:
                    continue  # Abort: human body / face

                # Cancellation Guard B: Forbidden Colors
                # E.g. in Red/White or Blue/White signs, substantial green foliage is forbidden
                if rule.primary_color in ["Red", "Blue"]:
                    crop_hsv = hsv[by:by+bh, bx:bx+bw]
                    green_mask = get_color_mask(crop_hsv, "Green")
                    green_ratio = cv2.countNonZero(green_mask) / float(bw * bh)
                    if green_ratio > 0.25:
                        continue  # Abort: tree foliage / nature background

                # Cancellation Guard C: Solidity / Hollow Sprawl
                solidity = area / float(bw * bh)
                if solidity < 0.30:
                    continue  # Abort: sprawling line / road pavement marking

                # Check color balance: both colors must be meaningfully present in the infected zone
                crop_pri = mask_pri[by:by+bh, bx:bx+bw]
                crop_sec = mask_sec[by:by+bh, bx:bx+bw]
                pri_ratio = cv2.countNonZero(crop_pri) / float(bw * bh)
                sec_ratio = cv2.countNonZero(crop_sec) / float(bw * bh)

                if pri_ratio < 0.03 or sec_ratio < 0.004 or (pri_ratio + sec_ratio) < 0.15:
                    continue

                bbox = BoundingBox(bx, by, bx + bw, by + bh)

                # Stage 4: OCR Number & Text Inspection on Surviving Plagues
                detected_text: Optional[SignTextResult] = None
                if self.enable_ocr and self.ocr_engine:
                    detected_text = self.ocr_engine.detect(crop_bgr)

                # Map to GTSRB & 100 Signs taxonomy
                pred_name, pred_id, pred_cat = self._infer_identity(
                    rule=rule,
                    aspect=aspect,
                    crop_bgr=crop_bgr,
                    detected_text=detected_text
                )

                conf = 0.70
                if detected_text and detected_text.confidence > 0.50:
                    conf = max(conf, detected_text.confidence)

                candidates.append((bbox, conf, pred_name, pred_id, pred_cat, f"Plague:{rule.combination_id}"))

        # Non-Maximum Suppression across all candidates
        candidates.sort(key=lambda c: c[1], reverse=True)
        final_results: List[Tuple[DetectionResult, ClassificationResult]] = []

        for bbox, conf, name, cid, cat, source in candidates:
            if not any(bbox.iou(kept_det.bbox) > 0.35 for kept_det, _ in final_results):
                det_res = DetectionResult(
                    bbox=bbox,
                    confidence=conf,
                    detector_label=name,
                    detector_class_id=cid
                )
                cls_res = ClassificationResult(
                    class_id=cid,
                    class_name=name,
                    confidence=conf,
                    category=cat
                )
                final_results.append((det_res, cls_res))

        return final_results

    def _infer_identity(
        self,
        rule: ColorCombinationRule,
        aspect: float,
        crop_bgr: np.ndarray,
        detected_text: Optional[SignTextResult]
    ) -> Tuple[str, int, SignCategory]:
        """Infer sign identity using color rule, shape geometry, and OCR output."""
        # Case 1: OCR detected a speed limit number (e.g. 50, 30, 70, 80, 100, 120)
        if detected_text and detected_text.sign_type == "SPEED_LIMIT" and detected_text.detected_number:
            speed = detected_text.detected_number
            speed_gtsrb_map = {
                20: (0, "Speed limit (20km/h)"),
                30: (1, "Speed limit (30km/h)"),
                50: (2, "Speed limit (50km/h)"),
                60: (3, "Speed limit (60km/h)"),
                70: (4, "Speed limit (70km/h)"),
                80: (5, "Speed limit (80km/h)"),
                100: (7, "Speed limit (100km/h)"),
                120: (8, "Speed limit (120km/h)"),
            }
            if speed in speed_gtsrb_map:
                cid, name = speed_gtsrb_map[speed]
                return name, cid, SignCategory.PROHIBITORY

        # Case 2: OCR detected 'STOP'
        if detected_text and detected_text.detected_word == "STOP":
            return "Stop", 14, SignCategory.PROHIBITORY

        # Case 3: OCR detected 'ZONE'
        if detected_text and detected_text.detected_word == "ZONE":
            return "Speed limit (30km/h)", 1, SignCategory.PROHIBITORY

        # Case 4: Color combination inference
        if rule.combination_id in ["RED_WHITE_BLACK", "RED_WHITE_SOLID"]:
            # Check shape: triangle (Danger) vs circle (Prohibitory) vs octagon (Stop)
            shape = self._estimate_shape(crop_bgr)
            if shape == "octagon":
                return "Stop", 14, SignCategory.PROHIBITORY
            elif shape == "triangle":
                return "General caution", 18, SignCategory.DANGER
            else:
                return "No vehicles", 15, SignCategory.PROHIBITORY

        elif rule.combination_id == "BLUE_WHITE":
            return "Ahead only", 35, SignCategory.MANDATORY

        elif rule.combination_id == "YELLOW_WHITE":
            return "Priority road", 12, SignCategory.OTHER

        elif rule.combination_id == "YELLOW_BLACK":
            return "General caution", 18, SignCategory.DANGER

        return "Traffic Sign", 0, SignCategory.OTHER

    def _estimate_shape(self, crop: np.ndarray) -> str:
        """Estimate polygon shape: circle, triangle, or octagon."""
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "circle"

        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        num_vertices = len(approx)

        if num_vertices == 3:
            return "triangle"
        elif num_vertices == 8:
            return "octagon"
        return "circle"


if __name__ == "__main__":
    detector = PlagueSecondaryDetector()

    # Test synthetic Stop sign
    canvas = np.full((300, 300, 3), 120, dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 60, (30, 30, 210), -1) # Red circle
    cv2.putText(canvas, "STOP", (110, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    results = detector.detect(canvas)
    print(f"Plague Detector found {len(results)} signs:")
    for det, cls in results:
        print(f"  -> [{cls.class_id}] {cls.class_name} (conf={cls.confidence:.2f}) at {det.bbox.to_xyxy()}")
