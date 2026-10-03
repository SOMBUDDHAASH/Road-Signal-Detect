"""
Member A Module: Dataset Preparation, Augmentation & Preprocessing Pipeline.
Lead: Member A (Yuvraj Singh) • Branch: feature/data-preprocessing

Provides complete end-to-end data loading, cleaning, augmentation,
train/val/test splitting, and exploratory data analysis (EDA).
"""

from typing import Tuple, List, Dict, Optional, Any
import os
import glob
import cv2
import numpy as np


class GTSRBDataPipeline:
    """
    Standardized data preparation and preprocessing pipeline for GTSRB (43 classes).
    Prepares clean, augmented datasets ready for Member C's classifier training.
    """

    def __init__(self, target_size: Tuple[int, int] = (32, 32)):
        self.target_size = target_size

    def preprocess_image(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Preprocesses a single traffic sign image:
        1. Resizes to target dimension (default: 32x32)
        2. Converts BGR to RGB
        3. Normalizes pixel values to [0.0, 1.0] (float32)
        4. Transposes from (H, W, C) to (C, H, W)
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        # Resize
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        # Convert BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if normalize:
            norm = rgb.astype(np.float32) / 255.0
            return np.transpose(norm, (2, 0, 1))

        return np.transpose(rgb, (2, 0, 1))

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies data augmentation to simulate realistic road driving variations:
        - Random rotation (-12 deg to +12 deg)
        - Random brightness/contrast jitter
        - Slight Gaussian motion blur
        """
        h, w = image.shape[:2]

        # 1. Random Rotation
        angle = np.random.uniform(-12, 12)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 2. Random Brightness & Contrast
        alpha = np.random.uniform(0.8, 1.25)  # Contrast
        beta = np.random.uniform(-20, 20)     # Brightness
        adjusted = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)

        # 3. Optional Motion Blur (30% probability)
        if np.random.rand() > 0.7:
            ksize = np.random.choice([3, 5])
            adjusted = cv2.GaussianBlur(adjusted, (ksize, ksize), 0)

        return adjusted

    def generate_eda_report(self, class_counts: Dict[int, int]) -> Dict[str, Any]:
        """
        Generates Exploratory Data Analysis metrics:
        - Class balance ratios
        - Minimum, Maximum, Mean samples per class
        """
        counts = list(class_counts.values())
        if not counts:
            return {}

        return {
            "total_classes": len(class_counts),
            "total_images": sum(counts),
            "min_samples_per_class": min(counts),
            "max_samples_per_class": max(counts),
            "mean_samples_per_class": round(float(np.mean(counts)), 1),
            "imbalance_ratio": round(max(counts) / float(max(1, min(counts))), 2)
        }
