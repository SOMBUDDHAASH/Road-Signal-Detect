"""
Unit tests for TemporalSignTracker and multi-frame ADAS state machine.
"""

import numpy as np
import pytest

from src.schema import BoundingBox, DetectionResult, ClassificationResult, PipelineDetection, SignCategory
from src.tracking.tracker import TemporalSignTracker, compute_iou


def test_compute_iou():
    boxA = BoundingBox(10, 10, 50, 50)
    boxB = BoundingBox(10, 10, 50, 50)
    # Exact overlap -> IoU = 1.0
    assert pytest.approx(compute_iou(boxA, boxB), 0.01) == 1.0

    # No overlap -> IoU = 0.0
    boxC = BoundingBox(60, 60, 100, 100)
    assert compute_iou(boxA, boxC) == 0.0


def test_temporal_tracker_registration_and_smoothing():
    tracker = TemporalSignTracker(smoothing_alpha=0.5, min_hits_to_confirm=1)

    # Frame 1: Detection at (10, 10, 50, 50)
    det1 = PipelineDetection(
        detection=DetectionResult(bbox=BoundingBox(10, 10, 50, 50), confidence=0.9),
        classification=ClassificationResult(class_id=2, class_name="Speed limit (50km/h)", confidence=0.9, category=SignCategory.PROHIBITORY)
    )

    tracked1 = tracker.update([det1])
    assert len(tracked1) == 1
    assert "Track #1" in tracked1[0].detection.detector_label
    assert tracked1[0].detection.bbox.to_xyxy() == (10, 10, 50, 50)

    # Frame 2: Slight movement to (12, 12, 52, 52)
    det2 = PipelineDetection(
        detection=DetectionResult(bbox=BoundingBox(12, 12, 52, 52), confidence=0.92),
        classification=ClassificationResult(class_id=2, class_name="Speed limit (50km/h)", confidence=0.92, category=SignCategory.PROHIBITORY)
    )

    tracked2 = tracker.update([det2])
    assert len(tracked2) == 1
    # Check that coordinate was smoothed (halfway between 10 and 12 = 11)
    assert tracked2[0].detection.bbox.x1 == 11
    assert tracked2[0].detection.bbox.y1 == 11


def test_tracker_adas_speed_limit_state():
    tracker = TemporalSignTracker(min_hits_to_confirm=2)

    det = PipelineDetection(
        detection=DetectionResult(bbox=BoundingBox(20, 20, 80, 80), confidence=0.95),
        classification=ClassificationResult(class_id=2, class_name="Speed limit (50km/h)", confidence=0.95, category=SignCategory.PROHIBITORY)
    )

    # Hit 1
    tracker.update([det])
    # Hit 2 (confirmed)
    tracker.update([det])

    assert tracker.current_speed_limit == "50 km/h"

    # End of speed limits sign (class 6)
    end_det = PipelineDetection(
        detection=DetectionResult(bbox=BoundingBox(20, 20, 80, 80), confidence=0.95),
        classification=ClassificationResult(class_id=6, class_name="End of speed limit (80km/h)", confidence=0.95, category=SignCategory.OTHER)
    )
    tracker.update([end_det])
    tracker.update([end_det])
    assert tracker.current_speed_limit == "No Limit"
