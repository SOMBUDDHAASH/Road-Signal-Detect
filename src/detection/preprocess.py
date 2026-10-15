"""Detection Preprocessing Utilities (Member B)
Includes letterbox resizing, CLAHE contrast enhancement, and bounding box cropping with boundary clamping.
"""

import cv2
import numpy as np
from src.config import DETECTION_IMG_SIZE


def letterbox(image: np.ndarray, target_size: int = DETECTION_IMG_SIZE, color=(114, 114, 114)) -> np.ndarray:
    """Resize image to target_size x target_size preserving aspect ratio with padding."""
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to L channel in LAB color space to enhance visibility in low light or fog."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def crop_sign(image: np.ndarray, bbox: list, padding: int = 4) -> np.ndarray:
    """Crop detected sign with margin padding and image boundary clamping."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    return crop.copy()
