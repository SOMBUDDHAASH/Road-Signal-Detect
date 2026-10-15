"""Data Preprocessing & Cleaning Module (Member A)
Performs ROI cropping, corrupt image removal, normalization, and resizing.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR, CLASSIFICATION_IMG_SIZE, NUM_CLASSES


def preprocess_image(image: np.ndarray, target_size: int = CLASSIFICATION_IMG_SIZE) -> np.ndarray:
    """Resize image to target_size x target_size and normalize pixel values to [0, 1]."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def clean_and_extract_crops(
    raw_dir: Path = DATA_RAW_DIR,
    target_size: int = CLASSIFICATION_IMG_SIZE,
    max_per_class: int = None
):
    """Clean raw GTSRB data, extract ROI crops, and save cleaned images."""
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_file = RESULTS_DIR / "cleaning_report.txt"

    train_csv = raw_dir / "Train.csv"
    train_dir = raw_dir / "Train"

    total_read = 0
    corrupt_count = 0
    cleaned_images = []
    cleaned_labels = []

    print("[PREPROCESS] Starting dataset cleaning and ROI cropping...")

    if train_csv.exists():
        df = pd.read_csv(train_csv)
        print(f"  Found Train.csv with {len(df)} annotations.")

        class_counts = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
            class_id = int(row["ClassId"])
            if max_per_class and class_counts.get(class_id, 0) >= max_per_class:
                continue

            rel_path = row["Path"]
            img_path = raw_dir / rel_path

            if not img_path.exists():
                corrupt_count += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                corrupt_count += 1
                continue

            total_read += 1

            # ROI coordinates
            x1, y1 = int(row["Roi.X1"]), int(row["Roi.Y1"])
            x2, y2 = int(row["Roi.X2"]), int(row["Roi.Y2"])

            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                corrupt_count += 1
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)

            cleaned_images.append(crop_resized)
            cleaned_labels.append(class_id)
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

    elif train_dir.exists():
        print(f"  Train.csv not found; scanning class subfolders in {train_dir}...")
        for class_id in range(NUM_CLASSES):
            class_folder = train_dir / str(class_id)
            if not class_folder.exists():
                continue

            count = 0
            for file in class_folder.iterdir():
                if file.suffix.lower() not in [".png", ".jpg", ".ppm"]:
                    continue
                if max_per_class and count >= max_per_class:
                    break

                img = cv2.imread(str(file))
                if img is None:
                    corrupt_count += 1
                    continue

                total_read += 1
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (target_size, target_size))

                cleaned_images.append(img_resized)
                cleaned_labels.append(class_id)
                count += 1

    else:
        print("[WARNING] Neither Train.csv nor Train/ found. Cannot process raw images.")
        return np.array([]), np.array([])

    X = np.array(cleaned_images, dtype=np.uint8)
    y = np.array(cleaned_labels, dtype=np.int64)

    # Save summary report
    with open(report_file, "w") as f:
        f.write("=== GTSRB Dataset Cleaning & Preprocessing Report ===\n")
        f.write(f"Total images read: {total_read}\n")
        f.write(f"Corrupt / unreadable images dropped: {corrupt_count}\n")
        f.write(f"Valid cleaned images retained: {len(X)}\n")
        f.write(f"Image resolution: {target_size}x{target_size}x3 (uint8 RGB)\n")
        f.write(f"Class count: {len(np.unique(y)) if len(y) > 0 else 0}\n")

    print(f"[PREPROCESS] Done! Retained {len(X)} images. Dropped {corrupt_count} corrupt.")
    print(f"  Cleaning report saved to: {report_file}")
    return X, y


if __name__ == "__main__":
    clean_and_extract_crops()
