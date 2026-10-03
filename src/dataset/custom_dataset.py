"""
Custom Dataset Manager.
Supports uploading custom dataset ZIPs (folder-per-class), extracting, validating,
and generating dynamic class mappings for custom regional traffic signs.
"""

from typing import Dict, List, Tuple, Optional
import os
import zipfile
import json
import glob
import shutil


class CustomDatasetManager:
    """
    Manages custom user datasets (e.g. Indian, US, or proprietary industrial signs).
    Accepts folder-per-class directories or zip files.
    """

    def __init__(self, storage_dir: str = "data/custom"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.classes_json_path = os.path.join(self.storage_dir, "classes.json")

    def unpack_zip(self, zip_path: str, dataset_name: str = "user_dataset") -> str:
        """Unpacks an uploaded ZIP dataset and returns the target directory."""
        target_dir = os.path.join(self.storage_dir, dataset_name)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)

        # Re-index classes
        self.index_dataset(target_dir)
        return target_dir

    def index_dataset(self, dataset_dir: str) -> Dict[int, str]:
        """
        Scans directory for subdirectories (each representing a class name).
        Generates class ID -> Class Name mapping and saves to classes.json.
        """
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        subdirs.sort()

        if not subdirs:
            # Check if there is a nested root directory
            items = os.listdir(dataset_dir)
            if len(items) == 1 and os.path.isdir(os.path.join(dataset_dir, items[0])):
                return self.index_dataset(os.path.join(dataset_dir, items[0]))

        class_map: Dict[int, str] = {}
        for idx, folder_name in enumerate(subdirs):
            clean_name = folder_name.replace("_", " ").title()
            class_map[idx] = clean_name

        with open(self.classes_json_path, "w") as f:
            json.dump({str(k): v for k, v in class_map.items()}, f, indent=2)

        return class_map

    def get_class_map(self) -> Dict[int, str]:
        """Loads active custom class mapping if available."""
        if os.path.exists(self.classes_json_path):
            with open(self.classes_json_path, "r") as f:
                data = json.load(f)
                return {int(k): v for k, v in data.items()}
        return {}

    def get_summary(self, dataset_dir: str) -> Dict[str, Any]:
        """Returns statistics on classes and image counts."""
        class_map = self.get_class_map()
        stats = {}
        total_images = 0
        for cid, cname in class_map.items():
            folder = os.path.join(dataset_dir, cname.replace(" ", "_").lower())
            if not os.path.exists(folder):
                # Try finding folder case-insensitively
                matches = [d for d in os.listdir(dataset_dir) if d.lower() == cname.replace(" ", "_").lower()]
                folder = os.path.join(dataset_dir, matches[0]) if matches else ""

            count = len(glob.glob(os.path.join(folder, "*.*"))) if folder and os.path.exists(folder) else 0
            stats[cname] = count
            total_images += count

        return {
            "num_classes": len(class_map),
            "total_images": total_images,
            "classes": class_map,
            "class_distribution": stats
        }
