"""
Mock and Heuristic Classifiers for testing and early integration.
Enables end-to-end pipeline execution before Member C completes CNN/MobileNetV3 training.
"""

from typing import List, Optional
import numpy as np
import cv2

from src.schema import ClassificationResult
from src.classification.base import BaseClassifier
from src.gtsrb_classes import get_class_name, get_sign_category


class MockClassifier(BaseClassifier):
    """
    Deterministic mock classifier for unit tests and interface compliance.
    """

    def __init__(self, default_class_id: int = 14, default_conf: float = 0.94):
        self.default_class_id = default_class_id
        self.default_conf = default_conf

    def classify(self, crop: np.ndarray) -> ClassificationResult:
        if crop is None or crop.size == 0:
            return ClassificationResult(
                class_id=-1,
                class_name="Invalid Crop",
                confidence=0.0
            )

        class_id = self.default_class_id
        class_name = get_class_name(class_id)
        category = get_sign_category(class_id)

        top_k = [
            (class_id, class_name, self.default_conf),
            ((class_id + 1) % 43, get_class_name((class_id + 1) % 43), round((1.0 - self.default_conf) * 0.7, 3)),
            ((class_id + 2) % 43, get_class_name((class_id + 2) % 43), round((1.0 - self.default_conf) * 0.3, 3)),
        ]

        return ClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=self.default_conf,
            category=category,
            top_k=top_k
        )


class ColorHeuristicClassifier(BaseClassifier):
    """
    Intelligent heuristic classifier that estimates GTSRB class based on color ratios:
    - Dominant Blue -> Mandatory (e.g. 38 Keep right, 35 Ahead only, 40 Roundabout)
    - Dominant Red Circular -> Prohibitory (e.g. 14 Stop, 1 Speed limit 30km/h, 17 No entry)
    - Dominant Red Triangular -> Danger (e.g. 18 General caution, 25 Road work)
    - Dominant Yellow/White -> Priority road (Class 12) or Yield (Class 13)
    """

    def classify(self, crop: np.ndarray) -> ClassificationResult:
        if crop is None or crop.size == 0:
            return ClassificationResult(class_id=-1, class_name="Invalid Crop", confidence=0.0)

        # Convert to HSV to measure dominant color hues
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_pixels = crop.shape[0] * crop.shape[1]

        # Red mask
        r1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
        red_ratio = (cv2.countNonZero(r1) + cv2.countNonZero(r2)) / total_pixels

        # Blue mask
        blue_mask = cv2.inRange(hsv, np.array([95, 80, 50]), np.array([130, 255, 255]))
        blue_ratio = cv2.countNonZero(blue_mask) / total_pixels

        # Yellow mask
        yellow_mask = cv2.inRange(hsv, np.array([15, 80, 70]), np.array([35, 255, 255]))
        yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels

        # Determine heuristic class
        if blue_ratio > 0.15 and blue_ratio > red_ratio:
            # Mandatory class (e.g. Class 38: Keep right or 35: Ahead only)
            class_id = 38
            confidence = min(0.95, 0.65 + blue_ratio)
        elif red_ratio > 0.18:
            # Prohibitory or Danger (Stop sign 14 or Yield 13 or Speed limit 30km/h 1)
            # Octagonal / circular check vs triangular check
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            num_vertices = 0
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                approx = cv2.approxPolyDP(largest_cnt, 0.04 * cv2.arcLength(largest_cnt, True), True)
                num_vertices = len(approx)

            if num_vertices == 3:
                class_id = 18 # General caution (Danger triangle)
            elif num_vertices == 8:
                class_id = 14 # Stop sign (Octagon)
            else:
                class_id = 1  # Speed limit (30km/h)
            confidence = min(0.96, 0.70 + red_ratio)
        elif yellow_ratio > 0.15:
            class_id = 12 # Priority road
            confidence = min(0.92, 0.65 + yellow_ratio)
        else:
            class_id = 13 # Yield / General
            confidence = 0.65

        class_name = get_class_name(class_id)
        category = get_sign_category(class_id)

        top_k = [
            (class_id, class_name, round(confidence, 3)),
            ((class_id + 1) % 43, get_class_name((class_id + 1) % 43), round((1.0 - confidence) * 0.7, 3)),
            ((class_id + 2) % 43, get_class_name((class_id + 2) % 43), round((1.0 - confidence) * 0.3, 3)),
        ]

        return ClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=round(confidence, 3),
            category=category,
            top_k=top_k
        )
