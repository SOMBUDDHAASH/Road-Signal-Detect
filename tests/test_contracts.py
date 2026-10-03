"""
Unit tests for data contracts, schemas, and GTSRB constants.
"""

import pytest
from src.schema import BoundingBox, DetectionResult, ClassificationResult, SignCategory
from src.gtsrb_classes import (
    GTSRB_CLASSES,
    GTSRB_CATEGORIES,
    get_class_name,
    get_sign_category,
    get_category_color_bgr
)


def test_bounding_box_init_and_properties():
    box = BoundingBox(10, 20, 110, 120)
    assert box.x1 == 10
    assert box.y1 == 20
    assert box.x2 == 110
    assert box.y2 == 120
    assert box.width == 100
    assert box.height == 100
    assert box.area == 10000
    assert box.center == (60, 70)
    assert box.to_xyxy() == (10, 20, 110, 120)
    assert box.to_xywh() == (10, 20, 100, 100)


def test_bounding_box_coordinate_reversal_guard():
    # If coordinates are given reversed (x2 < x1 or y2 < y1), it should automatically fix
    box = BoundingBox(150, 200, 50, 100)
    assert box.x1 == 50
    assert box.y1 == 100
    assert box.x2 == 150
    assert box.y2 == 200


def test_bounding_box_clamping():
    box = BoundingBox(-20, -10, 600, 700)
    clamped = box.clamp(max_width=500, max_height=400)
    assert clamped.x1 == 0
    assert clamped.y1 == 0
    assert clamped.x2 == 500
    assert clamped.y2 == 400


def test_bounding_box_padding():
    box = BoundingBox(50, 50, 150, 150) # width=100, height=100
    padded = box.pad(padding_ratio=0.10, max_width=300, max_height=300)
    assert padded.x1 == 40
    assert padded.y1 == 40
    assert padded.x2 == 160
    assert padded.y2 == 160


def test_gtsrb_classes_completeness():
    # Must have exactly 43 classes (0 to 42)
    assert len(GTSRB_CLASSES) == 43
    assert len(GTSRB_CATEGORIES) == 43
    for i in range(43):
        assert i in GTSRB_CLASSES
        assert len(GTSRB_CLASSES[i]) > 0
        assert i in GTSRB_CATEGORIES
        assert isinstance(GTSRB_CATEGORIES[i], SignCategory)


def test_gtsrb_helper_functions():
    assert get_class_name(14) == "Stop"
    assert get_sign_category(14) == SignCategory.PROHIBITORY
    assert get_class_name(38) == "Keep right"
    assert get_sign_category(38) == SignCategory.MANDATORY
    color = get_category_color_bgr(SignCategory.PROHIBITORY)
    assert isinstance(color, tuple) and len(color) == 3
