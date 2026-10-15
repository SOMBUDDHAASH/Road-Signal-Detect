"""Data Acquisition Module (Member A)
Supports direct path linking, extraction from zip, or automatic Kaggle download.
"""

import os
import shutil
import argparse
from pathlib import Path
from src.config import DATA_RAW_DIR

KAGGLE_DATASET_ID = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"


def download_or_link_dataset(source_path: str = None) -> Path:
    """Download GTSRB from Kaggle or copy/link from existing local directory."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    # 1. If explicit path is provided
    if source_path and os.path.exists(source_path):
        source = Path(source_path)
        print(f"[DATA] Ingesting GTSRB from local directory: {source}")

        # Check if source contains Train/ or Train.csv
        for item in ["Train", "Test", "Meta", "Train.csv", "Test.csv", "Meta.csv"]:
            src_item = source / item
            dst_item = DATA_RAW_DIR / item
            if src_item.exists() and not dst_item.exists():
                if src_item.is_dir():
                    shutil.copytree(src_item, dst_item)
                else:
                    shutil.copy2(src_item, dst_item)
                print(f"  Copied {item} -> {dst_item}")
        return DATA_RAW_DIR

    # 2. Check common known locations (e.g. Downloads/archive)
    user_downloads = Path.home() / "Downloads" / "archive"
    if user_downloads.exists() and (user_downloads / "Train.csv").exists():
        print(f"[DATA] Found local GTSRB archive at {user_downloads}. Copying to {DATA_RAW_DIR}...")
        return download_or_link_dataset(str(user_downloads))

    # 3. Download via kagglehub if available
    try:
        import kagglehub
        print(f"[DATA] Downloading GTSRB dataset via kagglehub ({KAGGLE_DATASET_ID})...")
        download_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
        print(f"[DATA] Downloaded to cache: {download_path}")
        return download_or_link_dataset(str(download_path))
    except Exception as e:
        print(f"[WARNING] kagglehub download failed or not configured: {e}")
        print("  Place the GTSRB dataset in data/raw/ manually or pass --path <path_to_extracted_archive>")
        return DATA_RAW_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or link GTSRB dataset")
    parser.add_argument("--path", type=str, default=None, help="Path to local extracted GTSRB dataset")
    args = parser.parse_args()

    out = download_or_link_dataset(args.path)
    print(f"[SUCCESS] Dataset ready at: {out}")
