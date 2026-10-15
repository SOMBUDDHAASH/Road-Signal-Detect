"""Data Augmentation & Class Balancing Module (Member A)
Performs class balancing and geometric/color augmentations without horizontal flips
(to prevent flipping directional traffic signs like Turn Left / Turn Right).
Saves train_balanced.npz and an augmentation preview chart.
"""

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import DATA_PROCESSED_DIR, RESULTS_DIR, NUM_CLASSES


def augment_single_image(img: np.ndarray) -> np.ndarray:
    """Apply random affine, rotation (-12 to +12 deg), brightness, and slight blur.
    CRITICAL: Never applies horizontal flip!
    """
    h, w = img.shape[:2]

    # 1. Random rotation between -12 and +12 degrees
    angle = np.random.uniform(-12, 12)
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 2. Random translation (-3 to +3 pixels)
    tx = np.random.randint(-3, 4)
    ty = np.random.randint(-3, 4)
    trans_mat = np.float32([[1, 0, tx], [0, 1, ty]])
    augmented = cv2.warpAffine(augmented, trans_mat, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3. Random brightness adjustment (0.8x to 1.2x)
    brightness_factor = np.random.uniform(0.8, 1.2)
    augmented = np.clip(augmented.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    # 4. Occasional slight Gaussian blur
    if np.random.rand() > 0.7:
        augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

    return augmented


def balance_and_augment(target_per_class: int = 500):
    """Load train.npz, augment under-represented classes to target_per_class,
    and save train_balanced.npz.
    """
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plots_dir = RESULTS_DIR / "plots"
    os.makedirs(plots_dir, exist_ok=True)

    train_path = DATA_PROCESSED_DIR / "train.npz"
    if not train_path.exists():
        print("[AUGMENT] train.npz not found. Running split_data first...")
        from src.data_pipeline.split_data import split_and_save_data
        split_and_save_data()

    data = np.load(train_path)
    X_train, y_train = data["X"], data["y"]

    print(f"[AUGMENT] Loaded {len(X_train)} training images. Target per class: {target_per_class}")

    balanced_X = []
    balanced_y = []

    preview_samples = []

    for class_id in range(NUM_CLASSES):
        class_indices = np.where(y_train == class_id)[0]
        class_images = X_train[class_indices]
        current_count = len(class_images)

        if current_count == 0:
            continue

        # Add all existing images
        balanced_X.extend(class_images)
        balanced_y.extend([class_id] * current_count)

        # Augment if under target
        needed = target_per_class - current_count
        if needed > 0:
            augmented_for_class = []
            for _ in range(needed):
                idx = np.random.randint(0, current_count)
                aug = augment_single_image(class_images[idx])
                augmented_for_class.append(aug)

            balanced_X.extend(augmented_for_class)
            balanced_y.extend([class_id] * needed)

            if len(preview_samples) < 9 and len(augmented_for_class) > 0:
                preview_samples.append((class_images[0], augmented_for_class[0], class_id))

    balanced_X = np.array(balanced_X, dtype=np.uint8)
    balanced_y = np.array(balanced_y, dtype=np.int64)

    # Shuffle balanced dataset
    indices = np.arange(len(balanced_X))
    np.random.shuffle(indices)
    balanced_X = balanced_X[indices]
    balanced_y = balanced_y[indices]

    out_path = DATA_PROCESSED_DIR / "train_balanced.npz"
    np.savez_compressed(out_path, X=balanced_X, y=balanced_y)

    print(f"[AUGMENT] Balanced dataset generated: {len(balanced_X)} images.")
    print(f"  Saved to: {out_path}")

    # Generate preview chart
    if preview_samples:
        fig, axes = plt.subplots(len(preview_samples), 2, figsize=(6, len(preview_samples) * 2))
        if len(preview_samples) == 1:
            axes = np.expand_dims(axes, 0)

        for i, (orig, aug, cid) in enumerate(preview_samples):
            axes[i, 0].imshow(orig)
            axes[i, 0].set_title(f"Class {cid} (Original)")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(aug)
            axes[i, 1].set_title(f"Class {cid} (Augmented)")
            axes[i, 1].axis("off")

        plt.tight_layout()
        preview_path = plots_dir / "augmentation_preview.png"
        plt.savefig(preview_path, dpi=120)
        plt.close()
        print(f"  Augmentation preview saved to: {preview_path}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500, help="Target images per class")
    args = parser.parse_args()

    balance_and_augment(target_per_class=args.target)
