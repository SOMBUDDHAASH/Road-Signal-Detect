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
