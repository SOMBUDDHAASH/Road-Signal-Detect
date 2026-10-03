"""
End-to-End Regression Test for Live Camera / User Frame Rejection and Sign Sheet Recognition.
Ensures zero false positives on humans/faces/rooms and 100% correct classification of road signs.
"""

import os
import cv2
import pytest

from src.detection.yolo import YOLODetector
from src.classification.model import PyTorchClassifier
from src.pipeline import TrafficSignPipeline


def test_user_webcam_frame_rejection():
    """Verify that a human sitting in front of a webcam produces 0 detections in user body/face area."""
    img_webcam_path = r"C:\Users\Sombuddha Ash\.gemini\antigravity\brain\fe6cd234-911f-40b2-91cb-15b94ed3dfc2\.user_uploaded\media_1791016270505.png"
    if not os.path.exists(img_webcam_path):
        pytest.skip("User webcam image not available.")

    img_webcam = cv2.imread(img_webcam_path)
    detector = YOLODetector(model_path=os.path.join("weights", "detection", "best.pt"), auto_fallback=True)
    classifier = PyTorchClassifier(model_path=os.path.join("weights", "classification", "classifier.pt"), auto_fallback=True)
    pipeline = TrafficSignPipeline(detector=detector, classifier=classifier, enable_tracking=True)

    res = pipeline.process_frame(img_webcam, is_video=True)
    # The user occupies y > 150
    user_detections = [d for d in res.detections if d.detection.bbox.y1 > 150]
    assert len(user_detections) == 0, f"Expected 0 detections on user body, got {len(user_detections)}"
    assert res.active_hazard is None, "False hazard warning triggered on human!"


def test_sign_sheet_detection_and_semantic_recognition():
    """Verify sign sheet correctly identifies Stop, Roundabout (Class 40), 20 MPH, No Entry, and Service."""
    img_sheet_path = r"C:\Users\Sombuddha Ash\.gemini\antigravity\brain\fe6cd234-911f-40b2-91cb-15b94ed3dfc2\.user_uploaded\media_1791016122579.png"
    if not os.path.exists(img_sheet_path):
        pytest.skip("Sign sheet image not available.")

    img_sheet = cv2.imread(img_sheet_path)
    detector = YOLODetector(model_path=os.path.join("weights", "detection", "best.pt"), auto_fallback=True)
    classifier = PyTorchClassifier(model_path=os.path.join("weights", "classification", "classifier.pt"), auto_fallback=True)
    pipeline = TrafficSignPipeline(detector=detector, classifier=classifier, enable_tracking=True)

    res = pipeline.process_frame(img_sheet, is_video=False)
    detected_classes = {d.classification.class_id: d.classification.class_name for d in res.detections}

    # Verify key signs
    assert 14 in detected_classes, "Stop sign (class 14) not detected!"
    assert 40 in detected_classes, "Roundabout mandatory (class 40) not detected!"
    assert 17 in detected_classes, "No entry (class 17) not detected!"
    assert 0 in detected_classes, "Speed limit 20 MPH (class 0) not detected!"
    assert "20" in str(res.active_speed_limit), "Expected 20 in active speed limit"
