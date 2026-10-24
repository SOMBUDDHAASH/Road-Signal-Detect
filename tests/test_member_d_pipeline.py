import os
import cv2
import pytest
import numpy as np

from src.pipeline import TrafficSignPipeline
from src.detection.shape_detector import ShapeContourDetector
from src.schema import BoundingBox, SignCategory


def test_member_d_shape_contour_detector_fallback():
    """Verify ShapeContourDetector identifies circular signs in graphic / compound plaques."""
    # Create synthetic test graphic: white plaque with a blue circular sign inside
    h, w = 600, 400
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    
    # Outer black border
    cv2.rectangle(canvas, (20, 20), (380, 580), (0, 0, 0), 4)
    # Blue circular traffic sign disc in upper region
    cv2.circle(canvas, (200, 180), 120, (180, 70, 20), -1)  # BGR blue
    # White diagonal arrow inside
    cv2.line(canvas, (160, 140), (240, 220), (255, 255, 255), 16)
    
    detector = ShapeContourDetector()
    detections = detector.detect(canvas)
    assert len(detections) >= 1
    
    # Check that candidate bounding box closely bounds the circular sign
    det = detections[0]
    assert 0.70 <= det.bbox.aspect_ratio <= 1.30
    assert det.detector_source == "Shape/Color Filter"


def test_member_d_pipeline_isolated_sign_recognition():
    """Verify end-to-end pipeline detects and recognizes isolated graphic signs via Member D fallback."""
    pipe = TrafficSignPipeline.create("production")
    
    # Create test image or load user image if present
    user_img_path = r"C:\Users\Sombuddha Ash\.gemini\antigravity\brain\4fe7d2cf-1e96-4f3b-b273-f6be1fbaa12c\.user_uploaded\media_1792829994446_2daa7b05.jpg"
    if os.path.exists(user_img_path):
        img = cv2.imread(user_img_path)
        result = pipe.process_frame(img, is_video=False)
        assert result.num_signs_detected >= 1
        det = result.detections[0]
        assert det.classification.class_id == 38  # Keep right
        assert "Keep right" in det.classification.class_name
        assert det.classification.confidence >= 0.85
        assert det.classification.category == SignCategory.MANDATORY


def test_member_d_canonical_samples_direct_crop_compatibility():
    """Verify canonical small crops continue to evaluate with 100% precision."""
    pipe = TrafficSignPipeline.create("production")
    sample_path = "data/samples/class_38_sample_1.png"
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        res = pipe.process_frame(img, is_video=False)
        assert res.num_signs_detected == 1
        assert res.detections[0].classification.class_id == 38
        assert res.detections[0].classification.confidence >= 0.95


def test_member_d_webcam_scene_zero_false_positives():
    """Verify live webcam room scene produces 0 false positive detections."""
    pipe = TrafficSignPipeline.create("production")
    webcam_img_path = r"C:\Users\Sombuddha Ash\.gemini\antigravity\brain\4fe7d2cf-1e96-4f3b-b273-f6be1fbaa12c\.user_uploaded\media_1792831010578_3e85326a.png"
    if os.path.exists(webcam_img_path):
        img = cv2.imread(webcam_img_path)
        res = pipe.process_frame(img, is_video=True)
        # Must suppress background room clutter and person clothing
        assert res.num_signs_detected == 0
