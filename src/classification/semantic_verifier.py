"""
Semantic and Physical Consistency Verifier for Traffic Signs.
Enforces real-world physical constraints (color palette, shape geometry,
and alphanumeric OCR) across GTSRB 43 classes and international road signs.
Maintained by Member D (Integration Lead).
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2

from src.schema import ClassificationResult, SignCategory


# GTSRB Class Metadata & Physical Archetypes
# Class groups:
# BLUE_MANDATORY: 33..40 (Mandatory blue circles with white symbols)
# RED_OCTAGON: 14 (STOP)
# RED_NO_ENTRY: 17 (No entry)
# RED_CIRCLE_SPEED: 0, 1, 2, 3, 4, 5, 7, 8 (Speed limits)
# RED_CIRCLE_PROHIBITORY: 9, 10, 15, 16 (Prohibitory signs)
# RED_TRIANGLE_DANGER: 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 (Danger signs)
# DERESTRICTION_GREY: 6, 32, 41, 42 (White/grey circles with diagonal slash)
# PRIORITY_DIAMOND: 12 (Yellow diamond priority road)
# YIELD_TRIANGLE: 13 (Inverted red triangle)

BLUE_MANDATORY_CLASSES = list(range(33, 41))
RED_TRIANGLE_DANGER_CLASSES = [11] + list(range(18, 32))
RED_CIRCLE_SPEED_CLASSES = {
    20: 0, 30: 1, 50: 2, 60: 3, 70: 4, 80: 5, 100: 7, 120: 8
}
SPEED_CLASS_TO_LIMIT = {
    0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120
}

GTSRB_CLASS_NAMES: Dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

GTSRB_CATEGORIES: Dict[int, SignCategory] = {
    **{i: SignCategory.PROHIBITORY for i in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 14, 15, 16, 17]},
    **{i: SignCategory.DANGER for i in [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]},
    **{i: SignCategory.MANDATORY for i in range(33, 41)},
    **{i: SignCategory.OTHER for i in [6, 12, 13, 32, 41, 42]}
}


class SemanticPhysicalVerifier:
    """
    Physical & Semantic Verification Engine for Traffic Signs.
    Prevents color hallucinations, rejects non-sign candidates (faces, bodies, rooms),
    and correctly fuses visual shape, color distribution, and OCR text.
    """

    def __init__(self):
        pass

    def analyze_crop_physics(self, crop: np.ndarray) -> Dict[str, float]:
        """Extract dominant color ratios, skin presence, edge variance, and geometry."""
        h, w = crop.shape[:2]
        total_px = float(max(1, h * w))

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Color masks
        red_mask = cv2.inRange(hsv, np.array([0, 65, 45]), np.array([10, 255, 255])) | \
                   cv2.inRange(hsv, np.array([170, 65, 45]), np.array([180, 255, 255]))
        blue_mask = cv2.inRange(hsv, np.array([100, 65, 45]), np.array([130, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 65, 45]), np.array([35, 255, 255]))
        white_mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 75, 255]))

        # Skin mask
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

        # Edge variance
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Check for centered horizontal white bar (No Entry signature)
        has_horizontal_bar = False
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask)
        for i in range(1, num_labels):
            bx, by, bw, bh, area = stats[i]
            aspect = bw / float(max(1, bh))
            if aspect >= 2.2 and bw >= (w * 0.40) and area >= (total_px * 0.05):
                # Check vertical centering (bar is near middle of circle)
                center_y = by + (bh / 2.0)
                if abs(center_y - (h / 2.0)) < (h * 0.25):
                    has_horizontal_bar = True
                    break

        return {
            "red_ratio": cv2.countNonZero(red_mask) / total_px,
            "blue_ratio": cv2.countNonZero(blue_mask) / total_px,
            "yellow_ratio": cv2.countNonZero(yellow_mask) / total_px,
            "white_ratio": cv2.countNonZero(white_mask) / total_px,
            "skin_ratio": cv2.countNonZero(skin_mask) / total_px,
            "lap_var": lap_var,
            "has_horizontal_bar": 1.0 if has_horizontal_bar else 0.0,
            "aspect_ratio": w / float(max(1, h))
        }

    def verify_and_correct(
        self,
        crop: np.ndarray,
        raw_class_id: int,
        raw_confidence: float,
        raw_probs: Optional[np.ndarray] = None,
        ocr_text: Optional[str] = None,
        ocr_number: Optional[int] = None,
        ocr_word: Optional[str] = None
    ) -> Optional[ClassificationResult]:
        """
        Applies physical consistency checks, negative rejection, and OCR overrides.
        Returns a verified ClassificationResult or None if rejected.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None

        physics = self.analyze_crop_physics(crop)

        # 1. Negative Filter: Reject human bodies, face crops, and flat room backgrounds
        if physics["skin_ratio"] > 0.35:
            return None
        if physics["lap_var"] < 12.0:
            return None

        sign_color_sum = physics["red_ratio"] + physics["blue_ratio"] + physics["yellow_ratio"]
        # If crop lacks sign colors and has low confidence, reject
        if sign_color_sum < 0.10 and raw_confidence < 0.65:
            return None

        # 1b. Direct High-Confidence GTSRB Stop Sign
        if raw_class_id == 14 and raw_confidence >= 0.70 and physics["red_ratio"] >= 0.15:
            return ClassificationResult(
                class_id=14,
                class_name="Stop",
                confidence=raw_confidence,
                category=SignCategory.PROHIBITORY
            )

        # 2. OCR Stop Sign Override
        if ocr_word == "STOP" or (ocr_text and "STOP" in ocr_text):
            return ClassificationResult(
                class_id=14,
                class_name="Stop",
                confidence=0.98,
                category=SignCategory.PROHIBITORY
            )

        # 3. OCR Yield Sign Override
        if ocr_word == "YIELD" or (ocr_text and "YIELD" in ocr_text):
            return ClassificationResult(
                class_id=13,
                class_name="Yield",
                confidence=0.96,
                category=SignCategory.OTHER
            )

        # 4. Yellow Speed Advisory / MUTCD Override (e.g. Yellow 20 MPH)
        if physics["yellow_ratio"] >= 0.28:
            if ocr_number in [15, 20, 25, 30, 35, 40, 45, 50] or (ocr_word in ["MPH", "M.P.H."]):
                speed_val = ocr_number if ocr_number else 20
                return ClassificationResult(
                    class_id=0,  # Map to GTSRB Class 0 / Speed limit
                    class_name=f"Speed limit ({speed_val} M.P.H.)",
                    confidence=0.92,
                    category=SignCategory.PROHIBITORY
                )
            # If yellow diamond with general caution / priority
            if raw_class_id not in [12, 18, 25]:
                # Force to Priority road (12) or Road work (25) or General caution (18)
                return ClassificationResult(
                    class_id=12,
                    class_name=GTSRB_CLASS_NAMES[12],
                    confidence=0.75,
                    category=GTSRB_CATEGORIES[12]
                )

        # 5. Blue Signs (Mandatory Circles vs. Accessible Service Rectangles)
        if physics["blue_ratio"] >= 0.25:
            # 5a. Vertical Rectangle with white symbol -> Accessible Service / Parking
            if physics["aspect_ratio"] <= 0.82 and physics["white_ratio"] >= 0.12:
                return ClassificationResult(
                    class_id=38,
                    class_name="Accessible Facility / Parking",
                    confidence=0.92,
                    category=SignCategory.OTHER
                )

            # 5b. Circular / Square Blue Mandatory (Roundabout, Ahead Only, Arrow Directions)
            if raw_probs is not None and len(raw_probs) == 43:
                blue_probs = {cid: float(raw_probs[cid]) for cid in BLUE_MANDATORY_CLASSES}
                best_cid = max(blue_probs, key=blue_probs.get)
                sum_p = sum(blue_probs.values())
                norm_conf = blue_probs[best_cid] / max(1e-6, sum_p)
                return ClassificationResult(
                    class_id=best_cid,
                    class_name=GTSRB_CLASS_NAMES[best_cid],
                    confidence=max(0.75, min(0.99, norm_conf)),
                    category=SignCategory.MANDATORY
                )
            else:
                if raw_class_id in BLUE_MANDATORY_CLASSES:
                    return ClassificationResult(
                        class_id=raw_class_id,
                        class_name=GTSRB_CLASS_NAMES[raw_class_id],
                        confidence=max(raw_confidence, 0.85),
                        category=SignCategory.MANDATORY
                    )
                # Default circular blue sign
                return ClassificationResult(
                    class_id=40,
                    class_name=GTSRB_CLASS_NAMES[40],
                    confidence=0.85,
                    category=SignCategory.MANDATORY
                )

        # 6. Red Circle Sign Consistency Gate (No Entry, Stop, Speed Limits, Prohibitory)
        if physics["red_ratio"] >= 0.22:
            # Check No Entry (horizontal white bar)
            if physics["has_horizontal_bar"] > 0.5:
                return ClassificationResult(
                    class_id=17,
                    class_name="No entry",
                    confidence=0.98,
                    category=SignCategory.PROHIBITORY
                )

            # Stop sign check (octagonal red or high red saturation)
            if physics["red_ratio"] > 0.50 and raw_class_id == 14:
                return ClassificationResult(
                    class_id=14,
                    class_name="Stop",
                    confidence=max(raw_confidence, 0.92),
                    category=SignCategory.PROHIBITORY
                )

            # Check if OCR found a speed limit number
            if ocr_number in RED_CIRCLE_SPEED_CLASSES:
                speed_cid = RED_CIRCLE_SPEED_CLASSES[ocr_number]
                return ClassificationResult(
                    class_id=speed_cid,
                    class_name=GTSRB_CLASS_NAMES[speed_cid],
                    confidence=0.95,
                    category=SignCategory.PROHIBITORY
                )

            # If raw classifier or YOLO predicted speed limit but NO digits were found by OCR,
            # this is a non-numeric prohibitory sign (e.g. Motorcycle / No vehicles)
            if raw_class_id in [0, 1, 2, 3, 4, 5, 7, 8] and ocr_number is None:
                return ClassificationResult(
                    class_id=15,
                    class_name="No vehicles / Motorcycles prohibited",
                    confidence=0.85,
                    category=SignCategory.PROHIBITORY
                )

        # 7. Red Triangle Danger Sign Consistency Gate
        if raw_class_id in RED_TRIANGLE_DANGER_CLASSES:
            if physics["red_ratio"] >= 0.10 or physics["yellow_ratio"] >= 0.25:
                return ClassificationResult(
                    class_id=raw_class_id,
                    class_name=GTSRB_CLASS_NAMES.get(raw_class_id, "Danger"),
                    confidence=raw_confidence,
                    category=SignCategory.DANGER
                )
            else:
                # Lacks red/yellow danger coloring -> Reject
                return None

        # 8. Accessible Parking / Facility Check
        if physics["blue_ratio"] >= 0.40 and physics["white_ratio"] >= 0.15:
            return ClassificationResult(
                class_id=38,
                class_name="Accessible Facility / Parking",
                confidence=0.88,
                category=SignCategory.OTHER
            )

        # 9. Strict Color Signature Presence:
        # A road sign MUST exhibit clear presence of red, blue, or yellow
        if sign_color_sum < 0.15:
            # Derestriction classes (6, 32, 41, 42) are white/grey with black slashes
            if raw_class_id in [6, 32, 41, 42] and physics["white_ratio"] >= 0.45:
                return ClassificationResult(
                    class_id=raw_class_id,
                    class_name=GTSRB_CLASS_NAMES[raw_class_id],
                    confidence=raw_confidence,
                    category=SignCategory.OTHER
                )
            # Otherwise, lack of road sign color indicates background/person/room noise -> Reject
            return None

        # 10. Fallback to raw prediction if confidence is sufficient and valid
        if raw_class_id in GTSRB_CLASS_NAMES and raw_confidence >= 0.35:
            return ClassificationResult(
                class_id=raw_class_id,
                class_name=GTSRB_CLASS_NAMES[raw_class_id],
                confidence=raw_confidence,
                category=GTSRB_CATEGORIES.get(raw_class_id, SignCategory.OTHER)
            )

        return None


# Global singleton verifier
_GLOBAL_VERIFIER: Optional[SemanticPhysicalVerifier] = None

def get_semantic_verifier() -> SemanticPhysicalVerifier:
    global _GLOBAL_VERIFIER
    if _GLOBAL_VERIFIER is None:
        _GLOBAL_VERIFIER = SemanticPhysicalVerifier()
    return _GLOBAL_VERIFIER
