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
