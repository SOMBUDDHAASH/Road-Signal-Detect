"""Unit & Integration tests for Member D Master Pipeline."""

import cv2
import numpy as np
import pytest
from src.integration.pipeline import TrafficSignPipeline
from src.integration.tracker import SignTracker
from src.schema import IntegratedDetection, DetectionResult, ClassificationResult, BoundingBox, SignCategory
from src.config import SAMPLES_DIR


def test_sign_tracker_continuity():
    tracker = SignTracker()

    box = BoundingBox(10, 10, 50, 50)
    det1 = IntegratedDetection(
        detection=DetectionResult(bbox=box, confidence=0.9, class_id=1, class_name="Speed limit (30km/h)"),
        classification=ClassificationResult(class_id=1, class_name="Speed limit (30km/h)", confidence=0.95, speed_limit=30),
        final_class_id=1,
        final_class_name="Speed limit (30km/h)",
        final_confidence=0.95,
    )

    # Frame 1
    out1 = tracker.update([det1], frame_idx=1)
    track_id1 = out1[0].track_id
    assert track_id1 is not None
    assert tracker.current_speed_limit == 30

    # Frame 2 with overlapping box
    box2 = BoundingBox(12, 11, 52, 51)
    det2 = IntegratedDetection(
        detection=DetectionResult(bbox=box2, confidence=0.88, class_id=1, class_name="Speed limit (30km/h)"),
        classification=ClassificationResult(class_id=1, class_name="Speed limit (30km/h)", confidence=0.92, speed_limit=30),
        final_class_id=1,
        final_class_name="Speed limit (30km/h)",
        final_confidence=0.92,
    )
    out2 = tracker.update([det2], frame_idx=2)
    assert out2[0].track_id == track_id1, "Track ID must remain continuous across frames"


def test_pipeline_smoke_run():
    pipeline = TrafficSignPipeline()
    sample_file = SAMPLES_DIR / "class_14_sample_1.png"
    if not sample_file.exists():
        pytest.skip("Sample file not found.")

    res = pipeline.process_image(str(sample_file))
    assert res.frame_index >= 1
    assert res.annotated_frame is not None
    assert res.inference_time_ms > 0
    assert len(res.detections) >= 1
    assert res.detections[0].final_class_id == 14
