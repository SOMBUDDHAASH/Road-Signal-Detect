"""
Script to extract sample test images and class metadata from archive.zip into data/
"""

import os
import zipfile
import shutil

ARCHIVE_PATH = r"C:\Users\Sombuddha Ash\Downloads\archive.zip"
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def extract_samples():
    if not os.path.exists(ARCHIVE_PATH):
        print(f"[Warning] Archive not found at {ARCHIVE_PATH}")
        return

    os.makedirs(os.path.join(TARGET_DIR, "samples"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "meta"), exist_ok=True)

    print(f"Reading from {ARCHIVE_PATH}...")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as z:
        namelist = z.namelist()
        
        # 1. Extract Meta.csv if present
        meta_csv = [f for f in namelist if f.lower().endswith("meta.csv")]
        if meta_csv:
            z.extract(meta_csv[0], TARGET_DIR)
            print(f"Extracted {meta_csv[0]}")

        # 2. Extract some sample test images (e.g. 30 diverse images)
        test_images = [f for f in namelist if f.startswith("Test/") and f.lower().endswith((".png", ".ppm", ".jpg"))]
        if not test_images:
            test_images = [f for f in namelist if f.startswith("test/") and f.lower().endswith((".png", ".ppm", ".jpg"))]

        print(f"Found {len(test_images)} test images. Extracting 30 sample images...")
        extracted_samples = 0
        for path in test_images[:30]:
            filename = os.path.basename(path)
            target_path = os.path.join(TARGET_DIR, "samples", filename)
            with z.open(path) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)
            extracted_samples += 1

        # 3. Extract sample meta icons (0.png to 42.png)
        meta_icons = [f for f in namelist if ("meta/" in f.lower() or "meta\\" in f.lower()) and f.lower().endswith(".png")]
        print(f"Found {len(meta_icons)} meta icon images. Extracting icons...")
        for path in meta_icons:
            filename = os.path.basename(path)
            if filename and filename.split(".")[0].isdigit():
                target_path = os.path.join(TARGET_DIR, "meta", filename)
                with z.open(path) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

    print(f"[Success] Extracted {extracted_samples} sample images into {os.path.join(TARGET_DIR, 'samples')}")


if __name__ == "__main__":
    extract_samples()
