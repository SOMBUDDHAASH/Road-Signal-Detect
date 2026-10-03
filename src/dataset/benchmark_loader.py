"""
GTSRB Benchmark Metadata & Test Ground Truth Loader.
Parses Meta.csv and Test.csv to provide ground-truth verification and benchmark evaluation.
"""

from typing import Dict, Optional, Any
import os
import csv
from dataclasses import dataclass

from src.gtsrb_classes import get_class_name, get_sign_category


@dataclass
class GroundTruthSign:
    filename: str
    width: int
    height: int
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int
    class_id: int
    class_name: str
    category: str
    shape_id: Optional[int] = None
    color_id: Optional[int] = None


class BenchmarkDataLoader:
    """
    Loads Meta.csv and Test.csv from data directory.
    Provides ground truth annotations for GTSRB benchmark samples.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.test_csv_path = os.path.join(data_dir, "Test.csv")
        self.meta_csv_path = os.path.join(data_dir, "Meta.csv")

        self.meta_info: Dict[int, Dict[str, Any]] = {}
        self.ground_truth_map: Dict[str, GroundTruthSign] = {}

        self._load_meta()
        self._load_test_csv()

    def _load_meta(self):
        if not os.path.exists(self.meta_csv_path):
            return

        with open(self.meta_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cid = int(row["ClassId"])
                    self.meta_info[cid] = {
                        "shape_id": int(row.get("ShapeId", -1)),
                        "color_id": int(row.get("ColorId", -1)),
                        "sign_id": row.get("SignId", "N/A"),
                        "path": row.get("Path", "")
                    }
                except (ValueError, KeyError):
                    continue

    def _load_test_csv(self):
        if not os.path.exists(self.test_csv_path):
            return

        with open(self.test_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    raw_path = row["Path"]
                    filename = os.path.basename(raw_path)
                    cid = int(row["ClassId"])
                    meta = self.meta_info.get(cid, {})

                    sign = GroundTruthSign(
                        filename=filename,
                        width=int(row["Width"]),
                        height=int(row["Height"]),
                        roi_x1=int(row["Roi.X1"]),
                        roi_y1=int(row["Roi.Y1"]),
                        roi_x2=int(row["Roi.X2"]),
                        roi_y2=int(row["Roi.Y2"]),
                        class_id=cid,
                        class_name=get_class_name(cid),
                        category=get_sign_category(cid).value,
                        shape_id=meta.get("shape_id"),
                        color_id=meta.get("color_id")
                    )
                    self.ground_truth_map[filename] = sign
                except (ValueError, KeyError):
                    continue

    def get_ground_truth(self, filename: str) -> Optional[GroundTruthSign]:
        """Look up ground truth for a test sample like '00000.png' or 'class_14_sample_1.png'."""
        base = os.path.basename(filename)
        if base in self.ground_truth_map:
            return self.ground_truth_map[base]

        # Support named samples: class_XX_sample_Y.png
        if base.startswith("class_") and "_sample_" in base:
            try:
                parts = base.split("_")
                cid = int(parts[1])
                if 0 <= cid < 43:
                    meta = self.meta_info.get(cid, {})
                    return GroundTruthSign(
                        filename=base,
                        width=32,
                        height=32,
                        roi_x1=0,
                        roi_y1=0,
                        roi_x2=32,
                        roi_y2=32,
                        class_id=cid,
                        class_name=get_class_name(cid),
                        category=get_sign_category(cid).value,
                        shape_id=meta.get("shape_id"),
                        color_id=meta.get("color_id")
                    )
            except (ValueError, IndexError):
                pass

        return None
