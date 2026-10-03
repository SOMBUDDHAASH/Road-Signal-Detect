"""
Unit & Integration Tests for Tsinghua-Tencent 100K (TT100K) Secondary Model.
Tests taxonomy, cross-domain ontology, model inference, consensus logic,
pipeline toggle behavior, visualizer HUD, and REST API endpoints.
Maintained by Member D (Integration & Pipeline Lead).
"""

import os
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from src.classification.tt100k_taxonomy import (
    TT100K_CLASSES,
    TT100K_DESCRIPTIONS,
    get_tt100k_name,
    map_tt100k_to_gtsrb,
    map_gtsrb_to_tt100k,
    evaluate_consensus
)
from src.classification.tt100k_model import (
    TT100KSecondaryClassifier,
    get_tt100k_classifier,
    create_and_export_tt100k_model
)
from src.schema import TT100KResult, PipelineDetection, DetectionResult, ClassificationResult, BoundingBox, SignCategory
from src.pipeline import TrafficSignPipeline
from src.detection.mock import MockDetector
from src.classification.mock import MockClassifier
from src.utils.visualizer import Visualizer
from api import app


def test_tt100k_taxonomy_completeness():
    """Verify official 221-class TT100K ontology specifications."""
    assert len(TT100K_CLASSES) == 221
    assert "pl50" in TT100K_CLASSES
    assert "ps" in TT100K_CLASSES
    assert "pg" in TT100K_CLASSES
    assert "w13" in TT100K_CLASSES
    assert "i1" in TT100K_CLASSES

    # Name resolutions
    assert "50km/h" in get_tt100k_name("pl50")
    assert "Stop" in get_tt100k_name("ps") or "Stop" in get_tt100k_name("p19")
    assert "Yield" in get_tt100k_name("pg")


def test_tt100k_cross_domain_bidirectional_mapping():
    """Verify bidirectional semantic mapping between TT100K and GTSRB."""
    # Speed limit 50: TT100K pl50 <-> GTSRB Class 2
    assert map_tt100k_to_gtsrb("pl50") == 2
    assert map_gtsrb_to_tt100k(2) == "pl50"

    # Stop sign: TT100K ps or p19 <-> GTSRB Class 14
    assert map_tt100k_to_gtsrb("ps") == 14
    assert map_tt100k_to_gtsrb("p19") == 14
    assert map_gtsrb_to_tt100k(14) == "ps"

    # Yield sign: TT100K pg <-> GTSRB Class 13
    assert map_tt100k_to_gtsrb("pg") == 13
    assert map_gtsrb_to_tt100k(13) == "pg"

    # Curve right: TT100K w13 <-> GTSRB Class 20
    assert map_tt100k_to_gtsrb("w13") == 20
    assert map_gtsrb_to_tt100k(20) == "w13"


def test_consensus_evaluation_logic():
    """Verify multi-domain agreement, category consensus, and domain discord."""
    # 1. Exact consensus
    is_con, note = evaluate_consensus(2, "pl50")
    assert is_con is True
    assert "CONSENSUS VERIFIED" in note

    # 2. Domain discord
    is_con, note = evaluate_consensus(2, "ps")
    assert is_con is False
    assert "DOMAIN DISCORD" in note

    # 3. Category consensus (e.g. speed limit family)
    is_con, note = evaluate_consensus(3, "pl40")  # pl40 has no direct GTSRB class, but is a speed limit
    assert is_con is True
    assert "CATEGORY CONSENSUS" in note


def test_tt100k_model_export_and_inference():
    """Verify model export, loading, and crop evaluation."""
    weights_path = os.path.join("weights", "classification", "tt100k_model.pt")
    assert os.path.exists(weights_path)

    classifier = TT100KSecondaryClassifier(model_path=weights_path)
    assert classifier.is_ready is True

    # Test dummy crop
    dummy = np.full((64, 64, 3), 128, dtype=np.uint8)
    res = classifier.analyze_crop(dummy, primary_gtsrb_id=2)
    assert isinstance(res, TT100KResult)
    assert res.class_code in TT100K_CLASSES
    assert res.confidence >= 0.0
    assert len(res.top_k) >= 1


def test_pipeline_tt100k_toggle_off():
    """Verify that when TT100K is toggled OFF, zero secondary overhead occurs and result is None."""
    pipeline = TrafficSignPipeline(
        detector=MockDetector(),
        classifier=MockClassifier(),
        enable_tracking=False,
        enable_tt100k=False
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame, is_video=False)

    assert result.num_signs_detected >= 1
    for d in result.detections:
        assert d.tt100k_result is None
    assert "tt100k_ms" not in result.latency_ms


def test_pipeline_tt100k_toggle_on():
    """Verify that when TT100K is toggled ON, secondary predictions and consensus are attached."""
    pipeline = TrafficSignPipeline(
        detector=MockDetector(),
        classifier=MockClassifier(),
        enable_tracking=False,
        enable_tt100k=True
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame, is_video=False)

    assert result.num_signs_detected >= 1
    assert "tt100k_ms" in result.latency_ms
    for d in result.detections:
        assert d.tt100k_result is not None
        assert isinstance(d.tt100k_result, TT100KResult)
        assert d.tt100k_result.class_code in TT100K_CLASSES


def test_visualizer_tt100k_annotations():
    """Verify that Visualizer draws TT100K badges without crashing."""
    viz = Visualizer()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    det = DetectionResult(bbox=BoundingBox(20, 20, 80, 80), confidence=0.9)
    cls_res = ClassificationResult(class_id=2, class_name="Speed limit (50km/h)", confidence=0.95, category=SignCategory.PROHIBITORY)
    tt_res = TT100KResult(
        class_code="pl50",
        class_name="Speed limit (50km/h)",
        confidence=0.97,
        mapped_gtsrb_id=2,
        is_consensus=True,
        consensus_note="CONSENSUS VERIFIED"
    )
    p_det = PipelineDetection(detection=det, classification=cls_res, crop=frame[20:80, 20:80], tt100k_result=tt_res)

    annotated = viz.annotate(frame, [p_det], latency_ms={"total_ms": 15.0, "tt100k_ms": 3.2}, fps=30.0)
    assert annotated.shape == frame.shape
    # Check that pixels were drawn (not all zeros)
    assert annotated.sum() > 0


def test_api_tt100k_endpoints():
    """Verify FastAPI endpoints for TT100K classes and predict with enable_tt100k."""
    client = TestClient(app)

    # 1. /tt100k/classes
    res = client.get("/tt100k/classes")
    assert res.status_code == 200
    data = res.json()
    assert data["total_classes"] == 221
    assert "pl50" in data["classes"]

    # 2. /predict with enable_tt100k=True
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", dummy_img)

    res = client.post(
        "/predict",
        files={"file": ("dummy.png", encoded.tobytes(), "image/png")},
        params={"mode": "mock", "enable_tt100k": True}
    )
    assert res.status_code == 200
    p_data = res.json()
    assert p_data["signs_detected"] >= 1
    # Check that tt100k field is populated
    assert p_data["detections"][0]["tt100k"] is not None
    assert p_data["detections"][0]["tt100k"]["class_code"] in TT100K_CLASSES
