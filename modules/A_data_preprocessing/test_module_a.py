"""
Unit tests for Member A (Data & Preprocessing Module).
Run: python -m pytest modules/A_data_preprocessing/test_module_a.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from modules.A_data_preprocessing.data_pipeline import GTSRBDataPipeline
from modules.A_data_preprocessing.dataset_downloader import check_dataset_status


def test_data_pipeline_initialization():
    pipeline = GTSRBDataPipeline(target_size=(32, 32))
    assert pipeline.target_size == (32, 32)


def test_preprocess_image():
    pipeline = GTSRBDataPipeline(target_size=(32, 32))
    dummy_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    processed = pipeline.preprocess_image(dummy_img, normalize=True)

    # Output should be (Channels, Height, Width) -> (3, 32, 32)
    assert processed.shape == (3, 32, 32)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_augment_image():
    pipeline = GTSRBDataPipeline()
    dummy_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

    augmented = pipeline.augment_image(dummy_img)

    assert augmented.shape == (32, 32, 3)
    assert augmented.dtype == np.uint8


def test_generate_eda_report():
    pipeline = GTSRBDataPipeline()
    sample_counts = {0: 180, 1: 1980, 2: 2010, 3: 1260}

    report = pipeline.generate_eda_report(sample_counts)

    assert report["total_classes"] == 4
    assert report["total_images"] == (180 + 1980 + 2010 + 1260)
    assert report["min_samples_per_class"] == 180
    assert report["max_samples_per_class"] == 2010
    assert report["imbalance_ratio"] > 1.0


def test_dataset_status_check():
    status = check_dataset_status()
    assert isinstance(status, dict)
    assert "is_ready" in status
