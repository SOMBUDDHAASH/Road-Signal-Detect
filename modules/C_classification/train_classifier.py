"""
Member C GTSRB Classifier Training Script.
Lead: Member C (Sneha Chakraborty) • Branch: feature/classification

Trains a Deep Convolutional Neural Network on the GTSRB dataset (43 classes).
Exports the optimized TorchScript model directly to:
  - modules/C_classification/weights/classifier.pt
  - weights/classification/classifier.pt
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train_custom import train_model, TrafficSignCNN


def load_gtsrb_dataset(data_dir: Path, samples_per_class: int = 150):
    """
    Loads GTSRB training images from data/Train directory or fallback search.
    """
    train_dir = data_dir / "Train"
    if not train_dir.exists():
        train_dir = data_dir / "train"

    images = []
    labels = []

    if train_dir.exists():
        print(f"[Data] Loading up to {samples_per_class} images per class from {train_dir}...")
        for cid in range(43):
            cdir = train_dir / str(cid)
            if not cdir.exists():
                continue
            fnames = list(cdir.glob("*.png")) + list(cdir.glob("*.ppm")) + list(cdir.glob("*.jpg"))
            count = 0
            for f in fnames:
                if count >= samples_per_class:
                    break
                img = cv2.imread(str(f))
                if img is not None:
                    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    norm = rgb.astype(np.float32) / 255.0
                    chw = np.transpose(norm, (2, 0, 1))
                    images.append(chw)
                    labels.append(cid)
                    count += 1
    else:
        # Fallback to archive.zip in Downloads
        archive_path = Path.home() / "Downloads" / "archive.zip"
        if archive_path.exists():
            import zipfile
            print(f"[Data] Reading from archive {archive_path}...")
            with zipfile.ZipFile(str(archive_path), "r") as z:
                class_files = {i: [] for i in range(43)}
                for name in z.namelist():
                    parts = name.split("/")
                    if len(parts) >= 3 and parts[0].lower() == "train" and parts[1].isdigit():
                        cid = int(parts[1])
                        if 0 <= cid < 43 and name.lower().endswith((".png", ".ppm", ".jpg")):
                            if len(class_files[cid]) < samples_per_class:
                                class_files[cid].append(name)

                for cid in range(43):
                    for fname in class_files[cid]:
                        with z.open(fname) as f:
                            file_bytes = np.frombuffer(f.read(), np.uint8)
                            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            if img is not None:
                                resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                                norm = rgb.astype(np.float32) / 255.0
                                chw = np.transpose(norm, (2, 0, 1))
                                images.append(chw)
                                labels.append(cid)

    if not images:
        raise RuntimeError(f"Could not find training images in {train_dir} or ~/Downloads/archive.zip")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"[Data] Successfully loaded {len(X)} samples across {len(np.unique(y))} classes. Shape: {X.shape}")
    return X, y


def run_training(epochs: int = 10, batch_size: int = 64, lr: float = 0.0015, samples: int = 120):
    data_dir = ROOT_DIR / "data"
    X, y = load_gtsrb_dataset(data_dir, samples_per_class=samples)

    # Train / Val Split (85% / 15%)
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.85 * len(X))
    X_train, y_train = X[indices[:split]], y[indices[:split]]
    X_val, y_val = X[indices[split:]], y[indices[split:]]

    out_mod_c = Path(__file__).resolve().parent / "weights" / "classifier.pt"
    out_mod_c.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Training] Starting PyTorch CNN training on {len(X_train)} samples, validating on {len(X_val)} samples...")
    t0 = time.time()
    val_acc, final_path = train_model(
        train_images=X_train,
        train_labels=y_train,
        val_images=X_val,
        val_labels=y_val,
        num_classes=43,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        output_path=str(out_mod_c)
    )

    print(f"\n[Success] Training complete in {time.time()-t0:.1f}s!")
    print(f"[Accuracy] Validation Accuracy: {val_acc:.2%}")
    print(f"[Saved] Module C weights saved to: {out_mod_c}")

    # Synchronize to root integration weights
    dest_root = ROOT_DIR / "weights" / "classification" / "classifier.pt"
    dest_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_mod_c, dest_root)
    print(f"[Deployed] Synchronized to integration pipeline: {dest_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Member C GTSRB Classifier Trainer")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0015, help="Learning rate")
    parser.add_argument("--samples", type=int, default=100, help="Samples per class to load")
    args = parser.parse_args()

    run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, samples=args.samples)
