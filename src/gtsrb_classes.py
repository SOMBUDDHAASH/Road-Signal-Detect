"""
GTSRB (German Traffic Sign Recognition Benchmark) 43 Class Definitions.
Includes human-readable names, visual categories, and styling tags.
"""

from typing import Dict, Tuple
from src.schema import SignCategory

GTSRB_CLASSES: Dict[int, str] = {
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
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

# Category assignment for all 43 classes
GTSRB_CATEGORIES: Dict[int, SignCategory] = {
    0: SignCategory.PROHIBITORY,
    1: SignCategory.PROHIBITORY,
    2: SignCategory.PROHIBITORY,
    3: SignCategory.PROHIBITORY,
    4: SignCategory.PROHIBITORY,
    5: SignCategory.PROHIBITORY,
    6: SignCategory.OTHER,
    7: SignCategory.PROHIBITORY,
    8: SignCategory.PROHIBITORY,
    9: SignCategory.PROHIBITORY,
    10: SignCategory.PROHIBITORY,
    11: SignCategory.DANGER,
    12: SignCategory.OTHER,
    13: SignCategory.OTHER,
    14: SignCategory.PROHIBITORY,
    15: SignCategory.PROHIBITORY,
    16: SignCategory.PROHIBITORY,
    17: SignCategory.PROHIBITORY,
    18: SignCategory.DANGER,
    19: SignCategory.DANGER,
    20: SignCategory.DANGER,
    21: SignCategory.DANGER,
    22: SignCategory.DANGER,
    23: SignCategory.DANGER,
    24: SignCategory.DANGER,
    25: SignCategory.DANGER,
    26: SignCategory.DANGER,
    27: SignCategory.DANGER,
    28: SignCategory.DANGER,
    29: SignCategory.DANGER,
    30: SignCategory.DANGER,
    31: SignCategory.DANGER,
    32: SignCategory.OTHER,
    33: SignCategory.MANDATORY,
    34: SignCategory.MANDATORY,
    35: SignCategory.MANDATORY,
    36: SignCategory.MANDATORY,
    37: SignCategory.MANDATORY,
    38: SignCategory.MANDATORY,
    39: SignCategory.MANDATORY,
    40: SignCategory.MANDATORY,
    41: SignCategory.OTHER,
    42: SignCategory.OTHER,
}

# BGR Color coding for visual bounding box and badge rendering
CATEGORY_COLORS_BGR: Dict[SignCategory, Tuple[int, int, int]] = {
    SignCategory.PROHIBITORY: (40, 40, 230),   # Vibrant Red
    SignCategory.DANGER: (30, 140, 255),       # Orange / Amber
    SignCategory.MANDATORY: (230, 130, 30),    # Vibrant Blue
    SignCategory.OTHER: (80, 200, 80),         # Fresh Green
}


def get_class_name(class_id: int) -> str:
    """Returns human-readable name for a GTSRB class ID."""
    return GTSRB_CLASSES.get(class_id, f"Unknown Sign ({class_id})")


def get_sign_category(class_id: int) -> SignCategory:
    """Returns sign category (Prohibitory, Danger, Mandatory, Other)."""
    return GTSRB_CATEGORIES.get(class_id, SignCategory.OTHER)


def get_category_color_bgr(category: SignCategory) -> Tuple[int, int, int]:
    """Returns (B, G, R) color for visualization."""
    return CATEGORY_COLORS_BGR.get(category, (200, 200, 200))
