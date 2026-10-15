"""Unit tests for Member C Classification Module."""

import cv2
import numpy as np
import pytest
from src.classification.keras_classifier import KerasSignClassifier
from src.config import SAMPLES_DIR, CLASS_NAMES


def test_keras_classifier_prediction_on_sample():
    sample_file = SAMPLES_DIR / "class_14_sample_1.png"
    if not sample_file.exists():
        pytest.skip("Sample class 14 file not found.")

    img = cv2.imread(str(sample_file))
    classifier = KerasSignClassifier()
    res = classifier.classify(img)

    assert res.class_id == 14, f"Expected class 14 (Stop), got {res.class_id}"
    assert "Stop" in res.class_name
    assert res.confidence > 0.90, f"Expected >90% confidence on clean Stop sign, got {res.confidence}"
    assert len(res.top_k) >= 3


def test_keras_classifier_batch():
    sample_file = SAMPLES_DIR / "class_14_sample_1.png"
    if not sample_file.exists():
        pytest.skip("Sample file not found.")

    img = cv2.imread(str(sample_file))
    classifier = KerasSignClassifier()
    batch_res = classifier.classify_batch([img, img])

    assert len(batch_res) == 2
    assert batch_res[0].class_id == 14
    assert batch_res[1].class_id == 14
