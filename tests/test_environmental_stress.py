"""
Aerospace-Standard Environmental Stress & Degradation Test Suite.
Maintained by Member D (Integration & Pipeline Lead).

Evaluates pipeline robustness against synthetic rain, fog, low-light night driving,
atmospheric haze, and verifies the Environmental Pre-Conditioner and Uncertainty Engine.
"""

import os
import glob
import cv2
import numpy as np
import pytest

from src.utils.environmental import get_environmental_conditioner, EnvironmentalConditioner
from src.utils.audio_alert import get_audio_transducer
from src.classification.model import PyTorchClassifier
from src.pipeline import TrafficSignPipeline
from src.dataset.benchmark_loader import BenchmarkDataLoader
from src.schema import PipelineDetection, DetectionResult, ClassificationResult, BoundingBox, SignCategory


def test_environmental_conditioner_night_enhancement():
    """Verify CLAHE and gamma expansion on synthetic night-time frame."""
    cond = get_environmental_conditioner()
    # Create dark frame with mean luminance ~35
    dark_frame = np.full((120, 120, 3), 35, dtype=np.uint8)
    telem = cond.analyze_scene(dark_frame)
    assert telem["is_night"] is True
    assert telem["mean_luminance"] < 50.0

    enhanced, telem_out = cond.auto_enhance(dark_frame)
    assert telem_out["was_enhanced"] is True
    assert any("night_gamma" in op for op in telem_out["applied_ops"])
    assert np.mean(enhanced) > np.mean(dark_frame)


def test_environmental_conditioner_fog_dehazing():
    """Verify dark-channel dehazing restores contrast under atmospheric scattering."""
    cond = get_environmental_conditioner()
    # Load canonical sample
    img = cv2.imread("data/samples/class_14_sample_1.png")
    if img is None:
        pytest.skip("Benchmark sample not found.")

    # Synthesize heavy atmospheric fog: I = J*t + A*(1-t)
    t = 0.50
    A = 220
    foggy = (img.astype(np.float32) * t + A * (1.0 - t)).astype(np.uint8)

    dehazed = cond.apply_dehaze(foggy)
    assert dehazed.shape == foggy.shape
    # Dehazed frame should have lower minimum channel and higher dynamic range
    assert np.std(dehazed) >= np.std(foggy)


def test_epistemic_uncertainty_metrics():
    """Verify Shannon entropy and confidence margin on clear vs ambiguous inputs."""
    clf = PyTorchClassifier(auto_fallback=True)
    img = cv2.imread("data/samples/class_14_sample_1.png")
    if img is None:
        pytest.skip("Benchmark sample not found.")

    res_clear = clf.classify(img)
    assert res_clear.class_id == 14
    assert res_clear.confidence > 0.85
    # Clear canonical sample should have low epistemic entropy and high margin
    assert res_clear.entropy < 2.0
    assert res_clear.margin > 0.70
    assert res_clear.is_ambiguous is False

    # Synthesize ambiguous uniform gray noise
    noise_patch = np.random.randint(90, 130, (32, 32, 3), dtype=np.uint8)
    res_noise = clf.classify(noise_patch)
    # Noise should either be rejected (class_id == -1) or have low confidence / ambiguity flag
    if res_noise.class_id >= 0:
        assert res_noise.confidence < 0.65 or res_noise.is_ambiguous is True or res_noise.entropy > 1.5


def test_audio_transducer_alerts_generation():
    """Verify ADASAudioTransducer generates correct acoustic chimes and spoken voice."""
    transducer = get_audio_transducer()

    # 1. Stop Sign -> chime_critical + 'Stop sign ahead'
    stop_det = PipelineDetection(
        detection=DetectionResult(BoundingBox(0, 0, 32, 32), 0.95),
        classification=ClassificationResult(14, "Stop", 0.98, SignCategory.PROHIBITORY)
    )
    alerts = transducer.get_pending_alerts([stop_det])
    assert len(alerts) == 1
    assert alerts[0]["type"] == "chime_critical"
    assert "Stop sign ahead" in alerts[0]["speech"]

    # 2. HTML payload generation
    html_payload = transducer.generate_html_audio_payload([stop_det], pending_alerts=alerts)
    assert html_payload is not None
    assert "AudioContext" in html_payload
    assert "speechSynthesis" in html_payload


def test_pipeline_with_environmental_enhancer():
    """Verify that TrafficSignPipeline executes environmental pre-conditioning without errors."""
    pipe = TrafficSignPipeline.create(mode="heuristic", enable_environmental_enhancer=True)
    dark_test = np.full((120, 120, 3), 30, dtype=np.uint8)
    res = pipe.process_frame(dark_test, is_video=False)

    assert res.environmental_telemetry is not None
    assert res.environmental_telemetry["is_night"] is True
    assert res.environmental_telemetry["was_enhanced"] is True
