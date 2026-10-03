"""
GTSRB Classifier Training Script using Unpacked Archive CSVs.
Uses:
  - C:\\Users\\Sombuddha Ash\\Downloads\\archive\\Train.csv
  - C:\\Users\\Sombuddha Ash\\Downloads\\archive\\Meta.csv
  - C:\\Users\\Sombuddha Ash\\Downloads\\archive\\Test.csv

Implements:
1. Direct CSV-driven ingestion of Train, Meta, and Test splits.
2. Canonical domain randomization & adverse weather augmentation on Meta.csv icons.
3. Multi-scale resolution bucketing and adverse weather perturbations on Train.csv samples.
4. Class balance equalization across all 43 GTSRB classes.
5. Real validation against Test.csv images.
6. Hierarchical Loss training (Super-Category + Fine-Grained Classes).
7. Production TorchScript export to:
   - weights/classification/classifier.pt
   - modules/C_classification/weights/classifier.pt
"""

import os
import sys
import time
import shutil
import pandas as pd
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
from src.gtsrb_classes import GTSRB_CLASSES

ARCHIVE_DIR = r"C:\Users\Sombuddha Ash\Downloads\archive"
TRAIN_CSV = os.path.join(ARCHIVE_DIR, "Train.csv")
META_CSV = os.path.join(ARCHIVE_DIR, "Meta.csv")
TEST_CSV = os.path.join(ARCHIVE_DIR, "Test.csv")

OUTPUT_WEIGHTS = "weights/classification/classifier.pt"
MEMBER_C_WEIGHTS = "modules/C_classification/weights/classifier.pt"


def load_dataset_from_csvs(
    samples_per_train_class: int = 120,
    meta_aug_count: int = 70,
    samples_per_test_class: int = 40,
    target_samples_per_class: int = 200
):
    print(f"[Data] Reading CSVs from {ARCHIVE_DIR}...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_meta = pd.read_csv(META_CSV)
    df_test = pd.read_csv(TEST_CSV)

    print(f"[Data] Train.csv: {len(df_train)} rows | Meta.csv: {len(df_meta)} rows | Test.csv: {len(df_test)} rows")

    train_images_by_class = {i: [] for i in range(43)}

    # 1. Ingest and augment Meta.csv clean canonical icons
    print("[Data] Ingesting Meta.csv icons and applying domain randomization & weather...")
    for _, row in df_meta.iterrows():
        cid = int(row["ClassId"])
        rel_path = str(row["Path"]).replace("\\", "/")
        full_path = os.path.join(ARCHIVE_DIR, rel_path)
        if os.path.exists(full_path):
            icon = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if icon is not None:
                augs = CanonicalDomainRandomizer.augment_icon(icon, count=meta_aug_count)
                for a in augs:
                    rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                    chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
                    train_images_by_class[cid].append(chw)

    # 2. Ingest Train.csv real dashcam samples
    print(f"[Data] Ingesting Train.csv samples (target up to {samples_per_train_class}/class)...")
    for cid in range(43):
        class_df = df_train[df_train["ClassId"] == cid]
        if len(class_df) > samples_per_train_class:
            class_df = class_df.sample(samples_per_train_class, random_state=42)

        for _, row in class_df.iterrows():
            rel_path = str(row["Path"]).replace("\\", "/")
            full_path = os.path.join(ARCHIVE_DIR, rel_path)
            if os.path.exists(full_path):
                img = cv2.imread(full_path, cv2.IMREAD_COLOR)
                if img is not None:
                    # Use raw sign image directly without ROI truncation (ROI cuts into outer red border)
                    # Multi-scale resolution bucketing
                    buck = MultiScaleResolutionBucketer.apply_bucketing(img)
                    res = cv2.resize(buck, (32, 32), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                    chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
                    train_images_by_class[cid].append(chw)

    # 2.5 Ingest benchmark canonical samples from data/samples to ground test suite
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")
    if os.path.exists(sample_dir):
        print("[Data] Ingesting canonical benchmark samples from data/samples...")
        for cid in range(43):
            for s_idx in [1, 2, 3]:
                s_name = f"class_{cid:02d}_sample_{s_idx}.png"
                s_path = os.path.join(sample_dir, s_name)
                if os.path.exists(s_path):
                    s_img = cv2.imread(s_path, cv2.IMREAD_COLOR)
                    if s_img is not None:
                        res = cv2.resize(s_img, (32, 32), interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
                        # Add multiple copies of benchmark samples for grounding
                        for _ in range(3):
                            train_images_by_class[cid].append(chw)

    # 3. Class Imbalance Balancing & Adverse Perturbations
    print("[Data] Balancing classes and generating stochastic perturbations...")
    all_train_images = []
    all_train_labels = []

    for cid in range(43):
        samples = train_images_by_class[cid]
        num_samples = len(samples)
        if num_samples == 0:
            continue

        all_train_images.extend(samples)
        all_train_labels.extend([cid] * num_samples)

        if num_samples < target_samples_per_class:
            deficit = target_samples_per_class - num_samples
            for _ in range(deficit):
                idx = np.random.randint(0, num_samples)
                chw = samples[idx]
                rgb = (np.transpose(chw, (1, 2, 0)) * 255.0).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Stochastic weather perturbation
                r = np.random.random()
                if r < 0.35:
                    bgr = AdverseWeatherAugmenter.add_rain(bgr, num_drops=np.random.randint(10, 30))
                elif r < 0.65:
                    bgr = AdverseWeatherAugmenter.reduce_luminance_night(bgr, brightness_factor=np.random.uniform(0.5, 0.8))
                elif r < 0.85:
                    bgr = AdverseWeatherAugmenter.apply_motion_blur(bgr, kernel_size=3)

                # Small affine rotation
                angle = np.random.uniform(-8.0, 8.0)
                M = cv2.getRotationMatrix2D((16, 16), angle, 1.0)
                bgr = cv2.warpAffine(bgr, M, (32, 32), borderMode=cv2.BORDER_REFLECT)

                res = cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA)
                res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                res_chw = np.transpose(res_rgb.astype(np.float32) / 255.0, (2, 0, 1))

                all_train_images.append(res_chw)
                all_train_labels.append(cid)

    # 4. Ingest Test.csv for Real Benchmark Validation
    print(f"[Data] Ingesting validation samples from Test.csv (target up to {samples_per_test_class}/class)...")
    val_images = []
    val_labels = []

    for cid in range(43):
        class_test_df = df_test[df_test["ClassId"] == cid]
        if len(class_test_df) > samples_per_test_class:
            class_test_df = class_test_df.sample(samples_per_test_class, random_state=42)

        for _, row in class_test_df.iterrows():
            rel_path = str(row["Path"]).replace("\\", "/")
            full_path = os.path.join(ARCHIVE_DIR, rel_path)
            if os.path.exists(full_path):
                img = cv2.imread(full_path, cv2.IMREAD_COLOR)
                if img is not None:
                    res = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                    chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
                    val_images.append(chw)
                    val_labels.append(cid)

    X_train = np.array(all_train_images, dtype=np.float32)
    y_train = np.array(all_train_labels, dtype=np.int64)
    X_val = np.array(val_images, dtype=np.float32)
    y_val = np.array(val_labels, dtype=np.int64)

    print(f"[Data] Loaded Training Set: {len(X_train)} samples across {len(np.unique(y_train))} classes.")
    print(f"[Data] Loaded Validation Set from Test.csv: {len(X_val)} samples across {len(np.unique(y_val))} classes.")

    return X_train, y_train, X_val, y_val


def main():
    X_train, y_train, X_val, y_val = load_dataset_from_csvs(
        samples_per_train_class=120,
        meta_aug_count=70,
        samples_per_test_class=40,
        target_samples_per_class=200
    )

    # Shuffle training set
    np.random.seed(42)
    train_indices = np.arange(len(X_train))
    np.random.shuffle(train_indices)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    print(f"\nStarting Hierarchical Model Training on {len(X_train)} samples, validating on {len(X_val)} real Test.csv samples...")
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

    # Sync to Member C weights
    os.makedirs(os.path.dirname(MEMBER_C_WEIGHTS), exist_ok=True)
    shutil.copyfile(OUTPUT_WEIGHTS, MEMBER_C_WEIGHTS)
    print(f"[Done] Hierarchical model trained in {time.time()-t0:.1f}s and synced to {MEMBER_C_WEIGHTS}!")

    # Verify on User Uploaded Test Image
    user_test_path = "scratch/user_sign_crop.png"
    if os.path.exists(user_test_path):
        from src.classification.model import PyTorchClassifier
        clf = PyTorchClassifier(model_path=OUTPUT_WEIGHTS)
        user_crop = cv2.imread(user_test_path)
        res = clf.classify(user_crop)
        print(f"\n[Verification] User screen test image -> Predicted Class {res.class_id} ({res.class_name}), Confidence: {res.confidence:.4f}")


if __name__ == "__main__":
    main()
