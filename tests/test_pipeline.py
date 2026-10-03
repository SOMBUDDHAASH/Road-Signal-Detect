"""
Integration tests for the unified TrafficSignPipeline.
"""

import os
import glob
import numpy as np
import cv2
import pytest

from src.pipeline import TrafficSignPipeline
from src.schema import PipelineResult


def test_pipeline_creation_modes():
    p_mock = TrafficSignPipeline.create(mode="mock")
    assert p_mock is not None

    p_heur = TrafficSignPipeline.create(mode="heuristic")
    assert p_heur is not None


def test_pipeline_empty_frame_raises():
    pipeline = TrafficSignPipeline.create(mode="mock")
    with pytest.raises(ValueError):
        pipeline.process_frame(np.array([]))


def test_pipeline_mock_run():
    pipeline = TrafficSignPipeline.create(mode="mock")
    frame = np.full((400, 500, 3), 100, dtype=np.uint8)

    result = pipeline.process_frame(frame)
    assert isinstance(result, PipelineResult)
    assert result.annotated_frame.shape == frame.shape
    assert result.num_signs_detected >= 1
    assert "total_ms" in result.latency_ms
    assert result.total_latency_ms >= 0.0

    det = result.detections[0]
    assert det.classification.class_id == 14
    assert det.crop is not None
    assert det.crop.shape[0] > 0 and det.crop.shape[1] > 0


def test_pipeline_on_dataset_samples():
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples")
    images = glob.glob(os.path.join(sample_dir, "*.png"))

    if not images:
        pytest.skip("No sample images in data/samples to test.")

    pipeline = TrafficSignPipeline.create(mode="heuristic")
    test_img = cv2.imread(images[0])
    assert test_img is not None

    result = pipeline.process_frame(test_img)
    assert isinstance(result, PipelineResult)
    assert result.annotated_frame.shape == test_img.shape
    assert result.latency_ms["detect_ms"] >= 0.0
    assert result.latency_ms["classify_ms"] >= 0.0


def test_pipeline_still_image_mode_bypasses_tracking():
    """Verify is_video=False allows detections to appear without multi-frame temporal confirmation."""
    pipeline = TrafficSignPipeline.create(mode="heuristic")
    frame = np.full((300, 300, 3), 120, dtype=np.uint8)
    # Draw a blue circle to simulate a mandatory sign
    cv2.circle(frame, (150, 150), 50, (200, 50, 20), -1)

    result = pipeline.process_frame(frame, conf_threshold=0.30, is_video=False)
    assert isinstance(result, PipelineResult)
    assert result.annotated_frame.shape == frame.shape


def test_pipeline_secondary_toggles_initialization():
    """Verify TrafficSignPipeline initializes and respects secondary toggles."""
    p_all_on = TrafficSignPipeline.create(
        mode="mock",
        enable_plague_detector=True,
        enable_ocr=True,
        enable_stage2_proposals=True,
        enable_semantic_verification=True
    )
    assert p_all_on.enable_plague_detector is True
    assert p_all_on.enable_ocr is True
    assert p_all_on.enable_stage2_proposals is True
    assert p_all_on.enable_semantic_verification is True
    assert p_all_on.secondary_detector is not None

    p_all_off = TrafficSignPipeline.create(
        mode="mock",
        enable_plague_detector=False,
        enable_ocr=False,
        enable_stage2_proposals=False,
        enable_semantic_verification=False
    )
    assert p_all_off.enable_plague_detector is False
    assert p_all_off.enable_ocr is False
    assert p_all_off.enable_stage2_proposals is False
    assert p_all_off.enable_semantic_verification is False
    assert p_all_off.secondary_detector is None


def test_pipeline_pure_gtsrb_mode_disables_secondary_fallback():
    """When secondary methods are toggled off, no secondary fallback should occur on empty primary results."""
    from src.detection.mock import MockDetector
    from src.classification.mock import MockClassifier

    class EmptyDetector(MockDetector):
        def detect(self, image, conf_threshold=0.5, **kwargs):
            return []

    empty_det = EmptyDetector()
    classifier = MockClassifier()

    # Test with plague enabled: empty primary triggers secondary search
    p_secondary_on = TrafficSignPipeline(
        detector=empty_det,
        classifier=classifier,
        enable_plague_detector=True,
        enable_tracking=False
    )
    assert p_secondary_on.enable_plague_detector is True

    # Test with plague disabled (Pure GTSRB mode): returns 0 detections
    p_secondary_off = TrafficSignPipeline(
        detector=empty_det,
        classifier=classifier,
        enable_plague_detector=False,
        enable_ocr=False,
        enable_stage2_proposals=False,
        enable_semantic_verification=False,
        enable_tracking=False
    )
    frame = np.full((300, 300, 3), 120, dtype=np.uint8)
    res_off = p_secondary_off.process_frame(frame)
    assert res_off.num_signs_detected == 0
    assert len(res_off.detections) == 0


def test_yolo_and_standalone_stage2_toggle():
    """Verify YOLODetector and StandaloneYOLODetector support enable_stage2_proposals."""
    from src.detection.yolo import YOLODetector
    from modules.B_detection.detector import StandaloneYOLODetector

    yolo = YOLODetector(enable_stage2_proposals=False)
    assert yolo.enable_stage2_proposals is False

    standalone = StandaloneYOLODetector(enable_stage2_proposals=False)
    assert standalone.enable_stage2_proposals is False

    # Dummy image inference with stage2 disabled
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    res1 = yolo.detect(dummy, enable_stage2_proposals=False)
    res2 = standalone.detect(dummy, enable_stage2_proposals=False)
    assert isinstance(res1, list)
    assert isinstance(res2, list)


