"""
Dataset Downloader and Extractor Utility for Member A.
Lead: Member A (Yuvraj Singh) • Branch: feature/data-preprocessing

This script extracts and prepares the GTSRB dataset for Member C (Classification)
and Member B (Detection). It searches for existing archive.zip downloads or
extracts local archives into standard data/ layout.
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
COMMON_ARCHIVE_PATHS = [
    Path.home() / "Downloads" / "archive.zip",
    Path.home() / "Downloads" / "gtsrb-german-traffic-sign.zip",
    Path(__file__).resolve().parent.parent.parent / "archive.zip"
]


def extract_archive(zip_path: str, target_dir: str = str(DEFAULT_DATA_DIR)) -> bool:
    """
    Extracts GTSRB dataset from a zip file into target_dir.
    """
    zip_path = Path(zip_path)
    target_dir = Path(target_dir)

    if not zip_path.exists():
        print(f"[Error] Zip archive not found: {zip_path}")
        return False

    print(f"[Info] Extracting {zip_path} -> {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        print(f"[Info] Archive contains {len(members)} entries.")
        zf.extractall(target_dir)

    print(f"[Success] Extraction complete. Files placed in: {target_dir}")
    return True


def check_dataset_status(target_dir: str = str(DEFAULT_DATA_DIR)) -> dict:
    """
    Checks if Train, Test, Meta datasets and CSVs are available.
    """
    p = Path(target_dir)
    status = {
        "train_dir_exists": (p / "Train").is_dir() or (p / "train").is_dir(),
        "test_dir_exists": (p / "Test").is_dir() or (p / "test").is_dir(),
        "meta_dir_exists": (p / "Meta").is_dir() or (p / "meta").is_dir(),
        "train_csv_exists": (p / "Train.csv").is_file(),
        "test_csv_exists": (p / "Test.csv").is_file(),
        "meta_csv_exists": (p / "Meta.csv").is_file()
    }
    status["is_ready"] = (
        (status["train_dir_exists"] or status["train_csv_exists"]) and
        (status["meta_csv_exists"] or status["test_csv_exists"])
    )
    return status


def auto_prepare(target_dir: str = str(DEFAULT_DATA_DIR)) -> bool:
    """
    Scans common download paths for GTSRB archive and extracts it.
    """
    status = check_dataset_status(target_dir)
    if status["is_ready"]:
        print(f"[Info] Dataset already present at: {target_dir}")
        for k, v in status.items():
            print(f"  - {k}: {v}")
        return True

    print("[Info] Dataset not fully detected. Searching for archive...")
    for candidate in COMMON_ARCHIVE_PATHS:
        if candidate.exists():
            print(f"[Info] Found archive at: {candidate}")
            return extract_archive(str(candidate), target_dir)

    print("[Warning] No local archive found. Please download GTSRB dataset from:")
    print("  https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    print(f"and place it at: {target_dir / 'archive.zip'}")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Member A Dataset Downloader & Extractor")
    parser.add_argument("--zip", type=str, help="Path to archive.zip", default=None)
    parser.add_argument("--target", type=str, help="Target extraction directory", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()

    if args.zip:
        extract_archive(args.zip, args.target)
    else:
        auto_prepare(args.target)
