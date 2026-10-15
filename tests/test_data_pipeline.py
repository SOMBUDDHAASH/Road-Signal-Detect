"""Unit tests for Member A Data Pipeline."""

import numpy as np
import pytest
from src.data_pipeline.preprocess import preprocess_image
from src.data_pipeline.augment import augment_single_image
from src.data_pipeline.data_loader import class_weights


def test_preprocess_image_shape_and_range():
    dummy = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    out = preprocess_image(dummy, target_size=32)

    assert out.shape == (32, 32, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_augmentation_does_not_flip():
    # Asymmetric image: left half black (0), right half white (255)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, 16:, :] = 255

    # Run augmentation 20 times, ensure the right side remains consistently brighter than left side
    for _ in range(20):
        aug = augment_single_image(img)
        left_mean = aug[:, :12, :].mean()
        right_mean = aug[:, 20:, :].mean()
        assert right_mean > left_mean, "Horizontal flip detected! Flips must not be applied."


def test_class_weights_calculation():
    # Class imbalance: class 0 has 10 samples, class 1 has 90 samples
    y = np.array([0] * 10 + [1] * 90)
    weights = class_weights(y)

    assert 0 in weights and 1 in weights
    assert weights[0] > weights[1], "Under-represented class must have higher weight"
