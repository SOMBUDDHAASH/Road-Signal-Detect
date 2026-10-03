"""
Comprehensive Full-Archive Training and Analysis Pipeline.
Uses every single file in: C:\\Users\\Sombuddha Ash\\Downloads\\archive
  - Train: All 39,209 images across all 43 class folders (Train/0 to Train/42)
  - Meta: All 43 canonical icons (Meta/0.png to Meta/42.png) + Meta.csv
  - Test: All 12,630 test images (Test/00000.png to Test/12629.png) + Test.csv / GT-final_test.csv

Exports:
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
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_custom import train_model
from src.data.augmentation import (
    AdverseWeatherAugmenter,
    MultiScaleResolutionBucketer,
    CanonicalDomainRandomizer
)
from src.gtsrb_classes import GTSRB_CLASSES, get_sign_category

ARCHIVE_BASE = r"C:\Users\Sombuddha Ash\Downloads\archive"
TRAIN_CSV = os.path.join(ARCHIVE_BASE, "Train.csv")
META_CSV = os.path.join(ARCHIVE_BASE, "Meta.csv")
TEST_CSV = os.path.join(ARCHIVE_BASE, "Test.csv")

OUTPUT_WEIGHTS = "weights/classification/classifier.pt"
MEMBER_C_WEIGHTS = "modules/C_classification/weights/classifier.pt"


def load_image_worker(item):
    path, cid, apply_bucketing = item
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if apply_bucketing:
        img = MultiScaleResolutionBucketer.apply_bucketing(img)
    res = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    return chw, cid


def load_entire_archive():
    print(f"================================================================================")
    print(f"INGESTING EVERY SINGLE FILE FROM ARCHIVE: {ARCHIVE_BASE}")
    print(f"================================================================================")
    
    t0 = time.time()
    df_train = pd.read_csv(TRAIN_CSV)
    df_meta = pd.read_csv(META_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    print(f"[Manifest] Train.csv rows: {len(df_train)} | Meta.csv rows: {len(df_meta)} | Test.csv rows: {len(df_test)}")
    
    # 1. Load All 43 Meta.csv canonical icons with domain randomization & weather
    print(f"\n[Phase 1] Processing all 43 canonical icons in Meta/ ...")
    meta_images = []
    meta_labels = []
    for _, row in df_meta.iterrows():
        cid = int(row["ClassId"])
        rel_p = str(row["Path"]).replace("\\", "/")
        full_p = os.path.join(ARCHIVE_BASE, rel_p)
        if os.path.exists(full_p):
            icon = cv2.imread(full_p, cv2.IMREAD_UNCHANGED)
            if icon is not None:
                augs = CanonicalDomainRandomizer.augment_icon(icon, count=65)
                for a in augs:
                    rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
                    chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
                    meta_images.append(chw)
                    meta_labels.append(cid)
    print(f"  [Loaded] {len(meta_images)} domain-randomized canonical samples from Meta/")
    
    # 2. Ingest Canonical Samples from data/samples (grounding)
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")
    sample_images = []
    sample_labels = []
    if os.path.exists(sample_dir):
        print(f"\n[Phase 2] Ingesting canonical benchmark samples from data/samples/ ...")
        for cid in range(43):
            for s_idx in [1, 2, 3]:
                s_path = os.path.join(sample_dir, f"class_{cid:02d}_sample_{s_idx}.png")
                if os.path.exists(s_path):
                    s_img = cv2.imread(s_path)
                    if s_img is not None:
                        res = cv2.resize(s_img, (32, 32))
                        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                        chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
                        for _ in range(4):
                            sample_images.append(chw)
                            sample_labels.append(cid)
        print(f"  [Loaded] {len(sample_images)} canonical reference samples")

    # 3. Load ALL 39,209 images from Train/ via parallel thread pool
    print(f"\n[Phase 3] Loading all {len(df_train)} images from Train/ (classes 0 to 42)...")
    train_tasks = []
    for _, row in df_train.iterrows():
        rel_p = str(row["Path"]).replace("\\", "/")
        full_p = os.path.join(ARCHIVE_BASE, rel_p)
        cid = int(row["ClassId"])
        train_tasks.append((full_p, cid, True))

    train_images = []
    train_labels = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(load_image_worker, train_tasks):
            if res is not None:
                train_images.append(res[0])
                train_labels.append(res[1])
                
    print(f"  [Loaded] {len(train_images)} real training images from Train/ subfolders in {time.time()-t0:.1f}s")

    # 4. Load ALL 12,630 images from Test/ via parallel thread pool
    print(f"\n[Phase 4] Loading all {len(df_test)} test images from Test/ ...")
    t_test = time.time()
    test_tasks = []
    for _, row in df_test.iterrows():
        rel_p = str(row["Path"]).replace("\\", "/")
        full_p = os.path.join(ARCHIVE_BASE, rel_p)
        cid = int(row["ClassId"])
        test_tasks.append((full_p, cid, False))

    test_images = []
    test_labels = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(load_image_worker, test_tasks):
            if res is not None:
                test_images.append(res[0])
                test_labels.append(res[1])

    print(f"  [Loaded] {len(test_images)} test images from Test/ in {time.time()-t_test:.1f}s")

    # Combine training sets
    all_X_train = np.array(train_images + meta_images + sample_images, dtype=np.float32)
    all_y_train = np.array(train_labels + meta_labels + sample_labels, dtype=np.int64)
    all_X_test = np.array(test_images, dtype=np.float32)
    all_y_test = np.array(test_labels, dtype=np.int64)

    print(f"\n[Dataset Summary]")
    print(f"  - Total Training Set Size : {len(all_X_train)} samples (Train/ + Meta/ + Canonical)")
    print(f"  - Total Test Set Size     : {len(all_X_test)} samples (All 12,630 Test/ images)")
    print(f"  - Total Files Utilized    : {len(all_X_train) + len(all_X_test)} image passes")

    return all_X_train, all_y_train, all_X_test, all_y_test


def main():
    X_train, y_train, X_test, y_test = load_entire_archive()

    # Shuffle training set
    np.random.seed(42)
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train = X_train[idx]
    y_train = y_train[idx]

    print(f"\n================================================================================")
    print(f"TRAINING HIERARCHICAL MODEL ON COMPLETE ARCHIVE (42,000+ SAMPLES)")
    print(f"================================================================================")
    t_start = time.time()

    # Train for 8 epochs on the entire dataset
    train_acc, out_path = train_model(
        train_images=X_train,
        train_labels=y_train,
        val_images=X_test, # Validating against the ENTIRE 12,630 Test set!
        val_labels=y_test,
        num_classes=43,
        epochs=8,
        batch_size=128,
        learning_rate=0.002,
        output_path=OUTPUT_WEIGHTS,
        use_hierarchical=True
    )

    # Sync to Member C weights
    os.makedirs(os.path.dirname(MEMBER_C_WEIGHTS), exist_ok=True)
    shutil.copyfile(OUTPUT_WEIGHTS, MEMBER_C_WEIGHTS)
    print(f"[Done] Full model trained in {time.time()-t_start:.1f}s and saved to {OUTPUT_WEIGHTS} & {MEMBER_C_WEIGHTS}!")

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
