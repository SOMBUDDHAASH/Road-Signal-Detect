"""
GTSRB Folder-Based Dataset Loader for Member C (Classification Lead).
Loads images from folder structure: data/Train/<class_id>/*.png (or .ppm / .jpg)
Maintains native OpenCV BGR color ordering and 32x32 normalization.
"""

import os
import cv2
import numpy as np
from typing import Tuple


def load_data(data_path: str = "data/Train", max_per_class: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Go through class folders 0 to 42, resize to (32, 32), normalize / 255.0.
    Returns:
      images: np.ndarray of shape (N, 32, 32, 3) in float32
      labels: np.ndarray of shape (N,) in int64
    """
    images = []
    labels = []

    if not os.path.exists(data_path):
        # Try alternate path relative to project root
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "Train")
        if os.path.exists(alt_path):
            data_path = alt_path

    # Go through class folders 0 to 42
    for class_id in range(43):
        class_path = os.path.join(data_path, str(class_id))
        if not os.path.exists(class_path):
            continue

        count = 0
        # Go through all images inside the class folder
        for image_name in os.listdir(class_path):
            if max_per_class > 0 and count >= max_per_class:
                break

            image_path = os.path.join(class_path, image_name)
            # Read image (keeps native BGR as Member C trained it)
            image = cv2.imread(image_path)
            # Skip if image could not be read
            if image is None:
                continue

            # Resize image to 32x32
            image = cv2.resize(image, (32, 32))
            # Normalize pixel values
            image = image / 255.0

            # Store image and label
            images.append(image)
            labels.append(class_id)
            count += 1

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Train"
    print(f"Loading dataset from: {path} ...")
    X, y = load_data(path, max_per_class=5)
    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)
