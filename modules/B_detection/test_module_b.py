"""
Unit tests for Member B (Detection Module).
Run: python -m pytest modules/B_detection/test_module_b.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.B_detection.detector import StandaloneYOLODetector
from src.schema import DetectionResult, BoundingBox


def test_detector_initialization():
    detector = StandaloneYOLODetector()
    assert detector is not None
    # Model should be ready if best.pt is present
    assert detector.is_ready is True


def test_detect_empty_or_none():
    detector = StandaloneYOLODetector()
    assert detector.detect(None) == []
    assert detector.detect(np.array([])) == []


def test_detect_dummy_image():
    detector = StandaloneYOLODetector()
    dummy = np.zeros((300, 300, 3), dtype=np.uint8)
    results = detector.detect(dummy, conf_threshold=0.5)

    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, DetectionResult)
        assert isinstance(r.bbox, BoundingBox)
        assert 0.0 <= r.confidence <= 1.0
        assert r.bbox.x1 >= 0 and r.bbox.y1 >= 0
        assert r.bbox.x2 <= 300 and r.bbox.y2 <= 300


def test_confidence_filtering():
    detector = StandaloneYOLODetector()
    dummy = np.zeros((300, 300, 3), dtype=np.uint8)
    # High threshold should return empty or subset
    res_high = detector.detect(dummy, conf_threshold=0.99)
    assert isinstance(res_high, list)
