"""Master Configuration for the Road Traffic Sign Detection & Recognition System.

Integrates parameters for:
- Member A (Data & Preprocessing)
- Member B (Detection - YOLOv8)
- Member C (Classification - CNN / Keras 3 / PyTorch)
- Member D (Integration & Pipeline)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODELS_DIR = PROJECT_ROOT / "models" if (PROJECT_ROOT / "models").exists() else WEIGHTS_DIR

# Data Paths
DATA_RAW_DIR = DATA_DIR / "raw" if (DATA_DIR / "raw" / "Train.csv").exists() else DATA_DIR
DATA_PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

# Model Paths (with robust fallbacks across weights/ and detection/models/)
YOLO_MODEL_PATH = WEIGHTS_DIR / "detection" / "best.pt"
if not YOLO_MODEL_PATH.exists() and (PROJECT_ROOT / "detection" / "models" / "best.pt").exists():
    YOLO_MODEL_PATH = PROJECT_ROOT / "detection" / "models" / "best.pt"

KERAS_MODEL_PATH = WEIGHTS_DIR / "classification" / "traffic_sign_model.keras"
if not KERAS_MODEL_PATH.exists() and (PROJECT_ROOT / "classification" / "models" / "traffic_sign_model.keras").exists():
    KERAS_MODEL_PATH = PROJECT_ROOT / "classification" / "models" / "traffic_sign_model.keras"

PYTORCH_MODEL_PATH = WEIGHTS_DIR / "classification" / "classifier.pt"

# Create necessary runtime directories
for path in [DATA_DIR, SAMPLES_DIR, RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

# Hardware & Inference Settings
KERAS_BACKEND = "torch"  # Fully compatible with PyTorch backend on Python 3.10-3.14
os.environ["KERAS_BACKEND"] = KERAS_BACKEND

# Preprocessing & Inference Dimensions
CLASSIFICATION_IMG_SIZE = 32  # 32x32 for GTSRB CNN
DETECTION_IMG_SIZE = 640       # Standard YOLOv8 input size
NUM_CLASSES = 43

# Thresholds
DEFAULT_DET_CONF = 0.25
DEFAULT_DET_IOU = 0.45
DEFAULT_CLS_CONF = 0.50
DEFAULT_BBOX_PADDING = 4

# 43 GTSRB Official Classes with categories and descriptions
CLASS_NAMES: Dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing (vehicles over 3.5t)",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve (left)",
    20: "Dangerous curve (right)",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows (right)",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all restrictions",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing (vehicles over 3.5t)",
}

# Sign Categories
CATEGORIES: Dict[str, List[int]] = {
    "PROHIBITORY": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17],
    "DANGER": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    "MANDATORY": [33, 34, 35, 36, 37, 38, 39, 40],
    "OTHER": [6, 11, 12, 13, 14, 32, 41, 42],
}


def get_category_for_class(class_id: int) -> str:
    for category, ids in CATEGORIES.items():
        if class_id in ids:
            return category
    return "OTHER"


# Speed Limit Mapping (extract numerical limit if sign is speed-related)
SPEED_LIMIT_MAP: Dict[int, int] = {
    0: 20,
    1: 30,
    2: 50,
    3: 60,
    4: 70,
    5: 80,
    7: 100,
    8: 120,
}
