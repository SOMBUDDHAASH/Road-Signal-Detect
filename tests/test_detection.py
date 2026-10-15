"""Unit tests for Member B Detection Module."""

import numpy as np
import pytest
from src.detection.yolo_detector import YOLODetector
from src.detection.shape_detector import ShapeContourDetector
from src.detection.preprocess import letterbox, enhance_contrast, crop_sign
from src.schema import BoundingBox


def test_letterbox_dimensions():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    lb = letterbox(img, target_size=640)
    assert lb.shape == (640, 640, 3)


def test_enhance_contrast():
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    enhanced = enhance_contrast(img)
    assert enhanced.shape == (100, 100, 3)
    assert enhanced.dtype == np.uint8


def test_crop_sign_clamping():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Box extending outside image boundaries
    crop = crop_sign(img, [-10, -10, 120, 120], padding=4)
    assert crop is not None
    assert crop.shape[0] == 100
    assert crop.shape[1] == 100


def test_shape_contour_detector():
    # Draw red circle on black background
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    import cv2
    cv2.circle(img, (100, 100), 30, (0, 0, 255), -1)  # BGR Red

    detector = ShapeContourDetector(min_area=100)
    dets = detector.detect(img)
    assert len(dets) >= 1
    box = dets[0].bbox
    assert box.width > 20 and box.height > 20
