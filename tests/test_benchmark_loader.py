"""
Unit tests for BenchmarkDataLoader using Meta.csv and Test.csv.
"""

import os
import pytest

from src.dataset.benchmark_loader import BenchmarkDataLoader


def test_benchmark_data_loader():
    loader = BenchmarkDataLoader()
    assert len(loader.meta_info) == 43

    # Check 00000.png from Test.csv
    gt0 = loader.get_ground_truth("00000.png")
    if gt0:
        assert gt0.class_id == 16
        assert gt0.class_name == "Vehicles over 3.5 metric tons prohibited"
        assert gt0.category == "Prohibitory"
        assert gt0.shape_id == 1
        assert gt0.color_id == 0
        assert gt0.roi_x1 == 6 and gt0.roi_y1 == 5


def test_all_43_classes_canonical_samples():
    """Verify that all 43 GTSRB classes have ground truth and canonical samples."""
    loader = BenchmarkDataLoader()
    for cid in range(43):
        gt = loader.get_ground_truth(f"class_{cid:02d}_sample_1.png")
        assert gt is not None, f"Missing ground truth for class {cid}"
        assert gt.class_id == cid, f"Mismatched class ID for class {cid}: got {gt.class_id}"
        assert gt.class_name != "Unknown / Unclassified"


def test_canonical_samples_classification_accuracy():
    """Verify that PyTorchClassifier correctly classifies >=95% of canonical benchmark samples."""
    import cv2
    import glob
    from src.classification.model import PyTorchClassifier

    clf = PyTorchClassifier(auto_fallback=True)
    loader = BenchmarkDataLoader()
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples")
    files = sorted(glob.glob(os.path.join(sample_dir, "class_*.png")))
    assert len(files) == 129, f"Expected 129 canonical samples, got {len(files)}"

    correct = 0
    for f in files:
        gt = loader.get_ground_truth(f)
        img = cv2.imread(f)
        res = clf.classify(img)
        if res.class_id == gt.class_id:
            correct += 1

    accuracy = correct / len(files)
    assert accuracy >= 0.95, f"Canonical benchmark accuracy too low: {accuracy*100:.2f}%"

