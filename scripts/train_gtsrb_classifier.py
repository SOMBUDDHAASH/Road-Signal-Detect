"""
Train a real GTSRB PyTorch classifier model directly using data from archive.zip.
Produces weights/classification/classifier.pt for production inference!
"""

import os
import sys
import zipfile
import io
import time
import numpy as np
import cv2
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_custom import train_model

ARCHIVE_PATH = r"C:\Users\Sombuddha Ash\Downloads\archive.zip"
OUTPUT_WEIGHTS = "weights/classification/classifier.pt"


def load_gtsrb_train_subset(samples_per_class: int = 80):
    print(f"Reading GTSRB training data from {ARCHIVE_PATH} (target: ~{samples_per_class} samples/class)...")
    images = []
    labels = []

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as z:
        # Group file names by class
        namelist = z.namelist()
        class_files = {i: [] for i in range(43)}

        for name in namelist:
            # Matches Train/0/xxx.png or train/0/xxx.png
            parts = name.split("/")
            if len(parts) >= 3 and parts[0].lower() == "train" and parts[1].isdigit():
                cid = int(parts[1])
                if 0 <= cid < 43 and name.lower().endswith((".png", ".ppm", ".jpg")):
                    if len(class_files[cid]) < samples_per_class:
                        class_files[cid].append(name)

        # Decode images
        for cid in range(43):
            file_list = class_files[cid]
            for fname in file_list:
                with z.open(fname) as f:
                    file_bytes = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is not None:
                        # Resize to 32x32 and normalize
                        resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        norm = rgb.astype(np.float32) / 255.0
                        chw = np.transpose(norm, (2, 0, 1))
                        images.append(chw)
                        labels.append(cid)

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"Loaded {len(X)} training images across 43 classes. Shape: {X.shape}")
    return X, y


def main():
    X, y = load_gtsrb_train_subset(samples_per_class=100)

    # Train / Val split (85% / 15%)
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.85 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Starting training on {len(X_train)} samples, validating on {len(X_val)} samples...")
    t0 = time.time()
    train_acc, out_path = train_model(
        train_images=X_train,
        train_labels=y_train,
        val_images=X_val,
        val_labels=y_val,
        num_classes=43,
        epochs=8,
        batch_size=64,
        learning_rate=0.0015,
        output_path=OUTPUT_WEIGHTS
    )
    print(f"[Done] Model trained in {time.time()-t0:.1f}s and saved to {out_path}!")


if __name__ == "__main__":
    main()
