"""
Unit tests for Member C (Classification Lead) integration & compatibility:
1. TensorFlowClassifier interface and BGR preprocessing
2. predict_sign() direct function interface
3. Folder-based data_loader.py
4. Automatic framework detection (PyTorch .pt vs TensorFlow .keras)
"""

import os
import numpy as np
import cv2
import pytest

from src.schema import ClassificationResult
from src.classification.tf_classifier import TensorFlowClassifier
from src.classification.model import load_classifier, PyTorchClassifier
from modules.C_classification.predict import predict_sign
from modules.C_classification.data_loader import load_data


def test_tf_classifier_interface_and_fallback():
    """Verify TensorFlowClassifier conforms to BaseClassifier contract and falls back safely if TF is absent."""
    classifier = TensorFlowClassifier(
        model_path="models/traffic_sign_model.keras",
        color_mode="bgr",
        auto_fallback=True
    )

    dummy_crop = np.full((50, 50, 3), 120, dtype=np.uint8)
    res = classifier.classify(dummy_crop)

    assert isinstance(res, ClassificationResult)
    assert hasattr(res, "class_id")
    assert hasattr(res, "confidence")
    assert hasattr(res, "class_name")
    assert hasattr(res, "category")


def test_tf_classifier_bgr_preprocessing():
    """Verify that BGR color mode preserves native OpenCV channel order without converting to RGB."""
    classifier = TensorFlowClassifier(
        model_path="models/traffic_sign_model.keras",
        color_mode="bgr",
        auto_fallback=True
    )

    # Pure Blue in BGR: (B=255, G=0, R=0)
    blue_crop = np.zeros((40, 40, 3), dtype=np.uint8)
    blue_crop[:, :, 0] = 255

    processed = classifier.preprocess(blue_crop)
    assert processed.shape == (32, 32, 3)
    # Channel 0 must remain 1.0 (Blue), Channels 1 and 2 must remain 0.0
    assert processed[0, 0, 0] == 1.0
    assert processed[0, 0, 1] == 0.0
    assert processed[0, 0, 2] == 0.0


def test_predict_sign_interface_on_sample():
    """Verify predict_sign() works as requested by Member C."""
    sample_path = os.path.join("data", "samples", "class_14_sample_1.png")
    if not os.path.exists(sample_path):
        pytest.skip("Sample image not found.")

    class_id, confidence = predict_sign(sample_path)
    assert isinstance(class_id, int)
    assert isinstance(confidence, float)
    assert class_id == 14  # Stop sign sample
    assert confidence >= 0.80


def test_predict_sign_with_numpy_array():
    """Verify predict_sign() works with a direct numpy crop from YOLO detection."""
    dummy_crop = np.full((64, 64, 3), 100, dtype=np.uint8)
    class_id, confidence = predict_sign(dummy_crop)
    assert isinstance(class_id, int)
    assert isinstance(confidence, float)


def test_load_classifier_auto_detection():
    """Verify load_classifier selects TensorFlowClassifier for .keras and PyTorchClassifier for .pt."""
    c_keras = load_classifier("models/traffic_sign_model.keras", auto_fallback=True)
    assert isinstance(c_keras, TensorFlowClassifier)

    c_pt = load_classifier("weights/classification/classifier.pt", auto_fallback=True)
    assert isinstance(c_pt, PyTorchClassifier)
