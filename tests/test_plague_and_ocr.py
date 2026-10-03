"""
Comprehensive Unit Tests for:
1. RoadSignOCREngine (Number detection & Alphabet/String detection)
2. 100 Most Common Road Signs Database
3. Road Sign Color Combinations Taxonomy
4. Plague Model Secondary Detector (Seed Discovery, Spreading, Immune Cancellation)
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw
import pytest

from src.dataset.common_signs_100 import COMMON_ROAD_SIGNS_100, get_road_sign_by_id, get_signs_by_category
from src.detection.color_taxonomy import ROAD_SIGN_COLOR_COMBINATIONS, get_color_mask, get_skin_mask
from src.detection.ocr_engine import RoadSignOCREngine, SignTextResult
from src.detection.plague_detector import PlagueSecondaryDetector
from src.pipeline import TrafficSignPipeline
from src.schema import PipelineResult, SignCategory


def test_100_common_signs_integrity():
    """Verify that exactly 100 international road signs are defined with complete metadata."""
    assert len(COMMON_ROAD_SIGNS_100) == 100

    ids = set()
    for sign in COMMON_ROAD_SIGNS_100:
        assert sign.id not in ids, f"Duplicate sign id: {sign.id}"
        ids.add(sign.id)
        assert len(sign.name) > 0
        assert sign.category in ["Priority", "Prohibitory", "Mandatory", "Danger", "Special", "Service", "Direction", "Supplementary"]
        assert len(sign.primary_color) > 0
        assert len(sign.shape) > 0

    # Test helpers
    stop_sign = get_road_sign_by_id(1)
    assert stop_sign is not None
    assert stop_sign.name == "Stop"
    assert stop_sign.has_text is True
    assert stop_sign.text_content == "STOP"

    prohibitory = get_signs_by_category("Prohibitory")
    assert len(prohibitory) >= 25


def test_color_combinations_taxonomy():
    """Verify standard color combination taxonomy rules."""
    assert len(ROAD_SIGN_COLOR_COMBINATIONS) >= 8

    rule_ids = [r.combination_id for r in ROAD_SIGN_COLOR_COMBINATIONS]
    assert "RED_WHITE_BLACK" in rule_ids
    assert "RED_WHITE_SOLID" in rule_ids
    assert "BLUE_WHITE" in rule_ids
    assert "YELLOW_BLACK" in rule_ids
    assert "GREEN_WHITE" in rule_ids
    assert "WHITE_BLACK" in rule_ids


def test_ocr_engine_speed_limit_numbers():
    """Verify OCR engine accurately detects speed limit numbers (50, 30, 70, 100)."""
    ocr = RoadSignOCREngine()

    for speed in [30, 50, 70, 100]:
        img = np.full((120, 120, 3), 255, dtype=np.uint8)
        cv2.circle(img, (60, 60), 55, (0, 0, 220), 8)
        p = Image.fromarray(img)
        draw = ImageDraw.Draw(p)
        draw.text((40 if speed < 100 else 32, 48), str(speed), fill=(0, 0, 0))
        arr = np.array(p)

        res = ocr.detect(arr)
        assert res is not None, f"OCR failed to detect speed {speed}"
        assert res.detected_number == speed
        assert res.sign_type == "SPEED_LIMIT"
        assert res.confidence >= 0.85


def test_ocr_engine_keywords():
    """Verify OCR engine accurately recognizes strings like STOP and ZONE."""
    ocr = RoadSignOCREngine()

    # Test STOP sign
    stop_img = np.full((120, 120, 3), 30, dtype=np.uint8)
    p = Image.fromarray(stop_img)
    draw = ImageDraw.Draw(p)
    draw.text((35, 48), "STOP", fill=(255, 255, 255))
    res_stop = ocr.detect(np.array(p))

    assert res_stop is not None
    assert res_stop.detected_word == "STOP"
    assert res_stop.sign_type == "STOP"
    assert res_stop.confidence >= 0.85

    # Test ZONE
    zone_img = np.full((120, 120, 3), 30, dtype=np.uint8)
    pz = Image.fromarray(zone_img)
    draw_z = ImageDraw.Draw(pz)
    draw_z.text((35, 48), "ZONE", fill=(255, 255, 255))
    res_zone = ocr.detect(np.array(pz))

    assert res_zone is not None
    assert res_zone.detected_word == "ZONE"
    assert res_zone.confidence >= 0.85


def test_plague_detector_seed_and_spread():
    """Verify Plague Model finds color seeds, spreads infection, and extracts signs."""
    detector = PlagueSecondaryDetector(enable_ocr=True)

    scene = np.full((350, 500, 3), 130, dtype=np.uint8)
    # Draw a synthetic Speed Limit 50 sign
    cv2.circle(scene, (150, 150), 50, (255, 255, 255), -1)
    cv2.circle(scene, (150, 150), 48, (0, 0, 220), 8)
    p = Image.fromarray(scene[100:200, 100:200])
    d = ImageDraw.Draw(p)
    d.text((38, 42), "50", fill=(0, 0, 0))
    scene[100:200, 100:200] = np.array(p)

    results = detector.detect(scene)
    assert len(results) >= 1

    det, cls = results[0]
    assert cls.class_id == 2  # GTSRB Class 2 is Speed limit (50km/h)
    assert "50" in cls.class_name
    assert cls.confidence >= 0.70


def test_plague_detector_immune_cancellation_on_skin():
    """Verify Plague Model immune cancellation aborts human skin tones and bodies."""
    detector = PlagueSecondaryDetector()

    scene = np.full((300, 300, 3), 130, dtype=np.uint8)
    # Red background (e.g. red jacket)
    cv2.rectangle(scene, (50, 120), (250, 260), (30, 30, 200), -1)
    # Human face (skin tone in BGR)
    cv2.circle(scene, (150, 80), 45, (110, 140, 210), -1)

    results = detector.detect(scene)
    assert len(results) == 0, f"Expected 0 detections due to immune cancellation, got {len(results)}"


def test_pipeline_secondary_fallback_integration():
    """Verify that when primary detector returns 0 signs, secondary detector recovers them."""
    # Use mock detector that always returns empty list to simulate primary failure
    class EmptyDetector:
        def detect(self, img, conf_threshold=0.5):
            return []

    class DummyClassifier:
        def classify_batch(self, crops):
            return []

    pipeline = TrafficSignPipeline(
        detector=EmptyDetector(),
        classifier=DummyClassifier(),
        enable_secondary_fallback=True
    )

    scene = np.full((300, 300, 3), 130, dtype=np.uint8)
    # Draw blue mandatory turn sign
    cv2.circle(scene, (150, 150), 50, (220, 70, 20), -1)
    cv2.arrowedLine(scene, (150, 175), (150, 125), (255, 255, 255), 6, tipLength=0.35)

    result = pipeline.process_frame(scene, conf_threshold=0.35, is_video=False)
    assert isinstance(result, PipelineResult)
    assert result.num_signs_detected >= 1
    assert result.detections[0].classification.category == SignCategory.MANDATORY
