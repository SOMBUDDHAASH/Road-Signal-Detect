"""
Road Sign Color Taxonomy & Pair Rules.
Defines standard international color combinations, HSV ranges,
adjacency pairs, and semantic associations for traffic sign detection.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2


@dataclass
class ColorCombinationRule:
    combination_id: str
    primary_color: str
    secondary_color: str
    accent_or_symbol_color: str
    semantic_meaning: str
    associated_categories: List[str]
    representative_examples: List[str]
    min_color_ratio: float = 0.12  # Minimum pixel fraction of both colors in the crop


# 10 Most Common Road Sign Color Combinations Worldwide
ROAD_SIGN_COLOR_COMBINATIONS: List[ColorCombinationRule] = [
    ColorCombinationRule(
        combination_id="RED_WHITE_BLACK",
        primary_color="Red",
        secondary_color="White",
        accent_or_symbol_color="Black",
        semantic_meaning="Prohibition, Restriction, and Danger Warnings (Vienna Convention)",
        associated_categories=["Prohibitory", "Danger", "Priority"],
        representative_examples=["Speed Limit (20-130)", "No Overtaking", "Warning Triangles", "Yield", "No Turn"]
    ),
    ColorCombinationRule(
        combination_id="RED_WHITE_SOLID",
        primary_color="Red",
        secondary_color="White",
        accent_or_symbol_color="White",
        semantic_meaning="Mandatory Stop and Critical Total Prohibitions",
        associated_categories=["Priority", "Prohibitory"],
        representative_examples=["Octagonal STOP Sign", "No Entry Circular Disc (White Bar)", "Wrong Way"]
    ),
    ColorCombinationRule(
        combination_id="BLUE_WHITE",
        primary_color="Blue",
        secondary_color="White",
        accent_or_symbol_color="White",
        semantic_meaning="Mandatory Actions, Positive Instructions, and High-Speed Motorway Guidance",
        associated_categories=["Mandatory", "Special", "Service"],
        representative_examples=["Ahead Only", "Turn Right/Left", "Roundabout", "Parking (P)", "Motorway Entrance", "Bus Lane"]
    ),
    ColorCombinationRule(
        combination_id="YELLOW_BLACK",
        primary_color="Yellow",
        secondary_color="Black",
        accent_or_symbol_color="Black",
        semantic_meaning="Physical Road Hazard Warnings (MUTCD Standard & European Temporary Construction)",
        associated_categories=["Danger", "Temporary"],
        representative_examples=["Sharp Curve Diamond", "Intersection Ahead", "Slippery Road", "Detour Ahead"]
    ),
    ColorCombinationRule(
        combination_id="YELLOW_WHITE",
        primary_color="Yellow",
        secondary_color="White",
        accent_or_symbol_color="None",
        semantic_meaning="Continuous Right-of-Way Priority Through All Crossings",
        associated_categories=["Priority"],
        representative_examples=["Priority Road Diamond (Yellow Center with White Outer Border)"]
    ),
    ColorCombinationRule(
        combination_id="GREEN_WHITE",
        primary_color="Green",
        secondary_color="White",
        accent_or_symbol_color="White",
        semantic_meaning="Directional Navigation, Highway Exits, Destination Mile Markers, and Permissive Movements",
        associated_categories=["Direction", "Service"],
        representative_examples=["Highway Direction Signs", "Exit Markers", "Mileposts", "Bike Route Green Signs"]
    ),
    ColorCombinationRule(
        combination_id="WHITE_BLACK",
        primary_color="White",
        secondary_color="Black",
        accent_or_symbol_color="Black",
        semantic_meaning="General Regulation, One-Way Streets, and End of Previous Restrictions",
        associated_categories=["Regulatory", "Prohibitory", "Special"],
        representative_examples=["End of Speed Limit", "End of All Restrictions (Diagonal Slash)", "One Way (US)", "Speed Limit (US White Rectangle)"]
    ),
    ColorCombinationRule(
        combination_id="ORANGE_BLACK",
        primary_color="Orange",
        secondary_color="Black",
        accent_or_symbol_color="Black",
        semantic_meaning="Active Highway Construction, Work Zones, Flag-Person, and Temporary Detours",
        associated_categories=["Temporary", "Work Zone"],
        representative_examples=["Road Work Ahead Diamond", "Detour 500 FT", "Flagger Ahead", "Lane Closed"]
    ),
    ColorCombinationRule(
        combination_id="BLUE_RED",
        primary_color="Blue",
        secondary_color="Red",
        accent_or_symbol_color="White",
        semantic_meaning="Standing, Waiting, and Parking Prohibitions",
        associated_categories=["Prohibitory"],
        representative_examples=["No Parking (Single Diagonal Red Stripe)", "No Stopping / Clearway (Crossed Red X)"]
    ),
    ColorCombinationRule(
        combination_id="BROWN_WHITE",
        primary_color="Brown",
        secondary_color="White",
        accent_or_symbol_color="White",
        semantic_meaning="Tourist Attractions, National Parks, Cultural Heritage, and Historical Landmarks",
        associated_categories=["Service", "Tourist"],
        representative_examples=["National Park Entrance", "Historical Site", "Campground", "Scenic Overlook"]
    )
]


# HSV Color Range Thresholds for Computer Vision Detection
COLOR_HSV_RANGES: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
    "Red": [
        # Red wraps around HSV hue (0-10 and 165-180)
        (np.array([0, 60, 50]), np.array([10, 255, 255])),
        (np.array([165, 60, 50]), np.array([180, 255, 255]))
    ],
    "Blue": [
        (np.array([95, 65, 45]), np.array([135, 255, 255]))
    ],
    "Yellow": [
        (np.array([16, 65, 60]), np.array([36, 255, 255]))
    ],
    "Orange": [
        (np.array([9, 85, 70]), np.array([22, 255, 255]))
    ],
    "Green": [
        (np.array([40, 50, 40]), np.array([85, 255, 255]))
    ],
    "White": [
        (np.array([0, 0, 160]), np.array([180, 55, 255]))
    ],
    "Black": [
        (np.array([0, 0, 0]), np.array([180, 255, 65]))
    ]
}


def get_color_mask(hsv_image: np.ndarray, color_name: str) -> np.ndarray:
    """Generate a binary mask for the specified color name in HSV space."""
    if color_name not in COLOR_HSV_RANGES:
        return np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_HSV_RANGES[color_name]:
        mask = cv2.inRange(hsv_image, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask


def get_skin_mask(bgr_image: np.ndarray) -> np.ndarray:
    """Detect human skin color in YCrCb color space to prevent false positives."""
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    # Standard human skin range in YCrCb
    return cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))


def is_skin_dominated(bgr_crop: np.ndarray, max_skin_ratio: float = 0.22) -> bool:
    """Return True if the image region is dominated by human skin tone."""
    if bgr_crop is None or bgr_crop.size == 0:
        return False
    mask = get_skin_mask(bgr_crop)
    ratio = cv2.countNonZero(mask) / float(bgr_crop.shape[0] * bgr_crop.shape[1])
    return ratio >= max_skin_ratio
