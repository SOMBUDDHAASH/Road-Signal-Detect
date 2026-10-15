"""Data Loader Module (Member A)
Handoff contract loader providing load_split, class_weights, and PyTorch / Keras loaders.
"""

import os
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from src.config import DATA_PROCESSED_DIR, NUM_CLASSES


def load_split(split_name: str = "train", normalize: bool = True, processed_dir: Path = DATA_PROCESSED_DIR):
    """Load a split ('train', 'train_balanced', 'val', or 'test') from .npz files.

    Returns:
        X: np.ndarray, float32 normalized in [0, 1] (if normalize=True) or uint8 in [0, 255]
        y: np.ndarray, int64 class labels 0-42
    """
    filename = f"{split_name}.npz" if not split_name.endswith(".npz") else split_name
    file_path = processed_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {file_path}. Run split_data.py or augment.py first."
        )

    data = np.load(file_path)
    X = data["X"]
    y = data["y"]

    if normalize:
        X = X.astype(np.float32) / 255.0

    return X, y


def class_weights(y: np.ndarray) -> dict:
    """Compute balanced class weights for training to handle class imbalance."""
    unique_classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y
    )
    return {int(i): float(w) for i, w in zip(unique_classes, weights)}


def get_torch_dataset(split_name: str = "train"):
    """Get PyTorch Dataset and DataLoader for the requested split."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    X, y = load_split(split_name, normalize=True)
    # Convert NHWC (OpenCV/NumPy) -> NCHW (PyTorch)
    X_torch = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
    y_torch = torch.tensor(y, dtype=torch.long)

    return TensorDataset(X_torch, y_torch)


def keras_dataset(split_name: str = "train", batch_size: int = 32, shuffle: bool = True):
    """Get a tf.data or numpy generator suitable for Keras model.fit."""
    X, y = load_split(split_name, normalize=True)
    return X, y
