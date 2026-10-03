"""
Unit tests for detector and classifier adapters.
"""

import numpy as np
import cv2
import pytest

from src.detection.mock import MockDetector, ColorContourDetector
from src.classification.mock import MockClassifier, ColorHeuristicClassifier
from src.schema import SignCategory


def test_mock_detector_default():
    detector = MockDetector(default_conf=0.90)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    detections = detector.detect(img, conf_threshold=0.5)

    assert len(detections) == 1
    det = detections[0]
    assert det.confidence == 0.90
    assert det.bbox.width > 0
    assert det.bbox.height > 0
    assert det.detector_label == "traffic_sign"


def test_mock_detector_confidence_filter():
    detector = MockDetector(default_conf=0.40)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # Threshold higher than confidence should yield empty list
    detections = detector.detect(img, conf_threshold=0.50)
    assert len(detections) == 0


def test_color_contour_detector_on_synthetic_sign():
    detector = ColorContourDetector(min_area=100)
    # Create image with a red circular sign
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    # Draw a filled red circle in center
    cv2.circle(img, (150, 150), 40, (0, 0, 255), -1)

    detections = detector.detect(img, conf_threshold=0.5)
    assert len(detections) >= 1
    det = detections[0]
    assert det.bbox.width > 30
    assert det.bbox.height > 30


def test_mock_classifier():
    classifier = MockClassifier(default_class_id=14, default_conf=0.95)
    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    result = classifier.classify(crop)

    assert result.class_id == 14
    assert result.class_name == "Stop"
    assert result.category == SignCategory.PROHIBITORY
    assert result.confidence == 0.95
    assert len(result.top_k) >= 3


def test_color_heuristic_classifier_blue():
    classifier = ColorHeuristicClassifier()
    # Create blue patch (Mandatory)
    crop = np.zeros((60, 60, 3), dtype=np.uint8)
    crop[:, :] = [255, 0, 0] # Pure blue in BGR
    result = classifier.classify(crop)

    assert result.category == SignCategory.MANDATORY
    assert result.class_id in [38, 35, 40]
    assert result.confidence > 0.5
