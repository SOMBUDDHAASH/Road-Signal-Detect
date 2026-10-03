"""
Unit tests for Member C (Classification Module).
Run: python -m pytest modules/C_classification/test_module_c.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.C_classification.classifier import StandaloneGTSRBClassifier
from src.schema import ClassificationResult, SignCategory


def test_classifier_initialization():
    classifier = StandaloneGTSRBClassifier()
    assert classifier is not None
    assert classifier.is_ready is True


def test_classify_empty_or_none():
    classifier = StandaloneGTSRBClassifier()
    res_none = classifier.classify(None)
    assert res_none.class_id == -1

    res_empty = classifier.classify(np.array([]))
    assert res_empty.class_id == -1


def test_preprocess_crop_shape():
    classifier = StandaloneGTSRBClassifier(img_size=32)
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    tensor_arr = classifier.preprocess(crop)

    # Expected: (1, 3, 32, 32) float32
    assert tensor_arr.shape == (1, 3, 32, 32)
    assert tensor_arr.dtype == np.float32


def test_classify_dummy_crop():
    classifier = StandaloneGTSRBClassifier()
    crop = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)

    res = classifier.classify(crop)
    assert isinstance(res, ClassificationResult)
    assert -1 <= res.class_id < 43
    assert 0.0 <= res.confidence <= 1.0
    assert isinstance(res.category, SignCategory)
