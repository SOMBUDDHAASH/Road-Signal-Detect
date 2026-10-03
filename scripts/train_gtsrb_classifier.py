"""
Calibrated Hierarchical GTSRB Classifier Training Script.
Maintained by Member D (Integration & Pipeline Lead).

Implements:
1. Adverse Weather Augmentation (rain, night luminance reduction, motion blur).
2. Multi-Scale Resolution Bucketing (16x16 to 128x128).
3. Class Imbalance Correction & Oversampling for minority classes.
4. Canonical Domain Randomization across all 43 classes.
5. Hierarchical Loss Training (Super-Category + Fine-Grained Classes).
6. Production TorchScript Export to weights/classification/classifier.pt.
"""

import os
import sys
import zipfile
import shutil
import time
import numpy as np
import cv2
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_custom import train_model
from src.data.augmentation import (
    AdverseWeatherAugmenter,
    MultiScaleResolutionBucketer,
    CanonicalDomainRandomizer
)

ARCHIVE_PATH = r"C:\Users\Sombuddha Ash\Downloads\archive.zip"
OUTPUT_WEIGHTS = "weights/classification/classifier.pt"
MEMBER_C_WEIGHTS = "modules/C_classification/weights/classifier.pt"


def load_augmented_gtsrb_dataset(
    samples_per_real_class: int = 100,
    meta_aug_count: int = 70,
    target_samples_per_class: int = 180
):
    print(f"[Data] Loading and augmenting GTSRB dataset from {ARCHIVE_PATH}...")
    images_by_class = {i: [] for i in range(43)}

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as z:
        namelist = z.namelist()

        # 1. Canonical Domain Randomization & Weather on Meta icons
        print("[Data] Generating canonical domain-randomized samples from Meta icons...")
        for cid in range(43):
            meta_name = f"Meta/{cid}.png"
            if meta_name in namelist:
                with z.open(meta_name) as f:
                    b = np.frombuffer(f.read(), np.uint8)
                    icon = cv2.imdecode(b, cv2.IMREAD_UNCHANGED)
                    if icon is not None:
                        augs = CanonicalDomainRandomizer.augment_icon(icon, count=meta_aug_count)
                        for a in augs:
                            rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                            chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
                            images_by_class[cid].append(chw)

        # 2. Real Train Samples from archive.zip
        print("[Data] Loading real dashcam samples from Train split...")
        class_files = {i: [] for i in range(43)}
        for name in namelist:
            parts = name.split("/")
            if len(parts) >= 3 and parts[0].lower() == "train" and parts[1].isdigit():
                cid = int(parts[1])
                if 0 <= cid < 43 and name.lower().endswith((".png", ".ppm", ".jpg")):
                    if len(class_files[cid]) < samples_per_real_class:
                        class_files[cid].append(name)

        for cid in range(43):
            for fname in class_files[cid]:
                with z.open(fname) as f:
                    b = np.frombuffer(f.read(), np.uint8)
                    raw_img = cv2.imdecode(b, cv2.IMREAD_COLOR)
                    if raw_img is not None:
                        # Multi-scale bucketing
                        buck = MultiScaleResolutionBucketer.apply_bucketing(raw_img)
                        res = cv2.resize(buck, (32, 32), interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
                        images_by_class[cid].append(chw)

    # 3. Class Imbalance Correction & Oversampling with Stochastic Adverse Perturbations
    print("[Data] Performing class balance correction and adverse weather perturbations...")
    all_images = []
    all_labels = []

    for cid in range(43):
        current_samples = images_by_class[cid]
        num_current = len(current_samples)

        if num_current == 0:
            continue

        # Add all existing samples
        all_images.extend(current_samples)
        all_labels.extend([cid] * num_current)

        # If minority class, oversample with weather & geometric perturbations
        if num_current < target_samples_per_class:
            deficit = target_samples_per_class - num_current
            for _ in range(deficit):
                pick_idx = np.random.randint(0, num_current)
                # Convert CHW back to BGR uint8
                chw = current_samples[pick_idx]
                rgb = (np.transpose(chw, (1, 2, 0)) * 255.0).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Stochastic perturbation
                dice = np.random.random()
                if dice < 0.30:
                    bgr = AdverseWeatherAugmenter.add_rain(bgr, num_drops=np.random.randint(10, 30))
                elif dice < 0.60:
                    bgr = AdverseWeatherAugmenter.reduce_luminance_night(bgr, brightness_factor=np.random.uniform(0.5, 0.8))
                elif dice < 0.80:
                    bgr = AdverseWeatherAugmenter.apply_motion_blur(bgr, kernel_size=3)

                # Slight rotation
                angle = np.random.uniform(-8.0, 8.0)
                M = cv2.getRotationMatrix2D((16, 16), angle, 1.0)
                bgr = cv2.warpAffine(bgr, M, (32, 32), borderMode=cv2.BORDER_REFLECT)

                res = cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA)
                res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                res_chw = np.transpose(res_rgb.astype(np.float32) / 255.0, (2, 0, 1))

                all_images.append(res_chw)
                all_labels.append(cid)

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    print(f"[Data] Final balanced dataset: {len(X)} samples across {len(np.unique(y))} classes. Shape: {X.shape}")
    return X, y


def main():
    X, y = load_augmented_gtsrb_dataset(
        samples_per_real_class=100,
        meta_aug_count=70,
        target_samples_per_class=180
    )

    # Train / Val Split (85% / 15%)
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(0.85 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Starting Hierarchical Training: {len(X_train)} train, {len(X_val)} val...")
    t0 = time.time()
    train_acc, out_path = train_model(
        train_images=X_train,
        train_labels=y_train,
        val_images=X_val,
        val_labels=y_val,
        num_classes=43,
        epochs=12,
        batch_size=64,
        learning_rate=0.002,
        output_path=OUTPUT_WEIGHTS,
        use_hierarchical=True
    )

    # Sync to Member C weights path as well
    os.makedirs(os.path.dirname(MEMBER_C_WEIGHTS), exist_ok=True)
    shutil.copyfile(OUTPUT_WEIGHTS, MEMBER_C_WEIGHTS)
    print(f"[Done] Hierarchical model trained in {time.time()-t0:.1f}s and synced to {MEMBER_C_WEIGHTS}!")


if __name__ == "__main__":
    main()
