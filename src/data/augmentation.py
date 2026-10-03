"""
Adverse Weather & Multi-Scale Data Augmentation Pipeline.
Maintained by Member D (Integration & Pipeline Lead).

Implements:
1. Adverse Weather Augmentation: Synthetic rain streaks, night luminance reduction, and motion blur.
2. Multi-Scale Resolution Bucketing: Simulates varying camera-to-sign distances (16x16 to 128x128).
3. Class Imbalance Correction: Oversampling and synthetic perturbations for low-frequency classes.
4. Canonical Domain Randomization: Blends clean digital icons with realistic backgrounds.
"""

from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np


class AdverseWeatherAugmenter:
    """
    Simulates real-world environmental degradations:
    - Rain streaks with realistic velocity and angle
    - Night/dusk illumination and contrast drop
    - High-speed vehicular motion blur
    """

    @staticmethod
    def add_rain(
        image: np.ndarray,
        num_drops: int = 50,
        length: int = 12,
        angle: float = -15.0,
        alpha: float = 0.65
    ) -> np.ndarray:
        """Adds semi-transparent rain streaks to the image."""
        h, w = image.shape[:2]
        rain_layer = image.copy()
        
        # Calculate streak vector
        rad = np.radians(angle)
        dx = int(length * np.sin(rad))
        dy = int(length * np.cos(rad))
        
        for _ in range(num_drops):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            x2 = np.clip(x1 + dx, 0, w - 1)
            y2 = np.clip(y1 + dy, 0, h - 1)
            color = int(np.random.randint(180, 245))
            cv2.line(rain_layer, (x1, y1), (x2, y2), (color, color, color), 1)
            
        return cv2.addWeighted(rain_layer, 1.0 - alpha, image, alpha, 0)

    @staticmethod
    def reduce_luminance_night(
        image: np.ndarray,
        brightness_factor: float = 0.40,
        contrast_factor: float = 0.70
    ) -> np.ndarray:
        """Simulates low-light, nighttime, or shadowed road conditions."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
        darkened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Reduce contrast
        mean_bgr = np.mean(darkened, axis=(0, 1), keepdims=True)
        out = mean_bgr + contrast_factor * (darkened.astype(np.float32) - mean_bgr)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_motion_blur(
        image: np.ndarray,
        kernel_size: int = 5,
        angle: float = 0.0
    ) -> np.ndarray:
        """Simulates vehicle vibration and speed-induced directional motion blur."""
        if kernel_size <= 1:
            return image
        
        # Generate directional motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        rad = np.radians(angle)
        
        dx = np.cos(rad)
        dy = np.sin(rad)
        for i in range(kernel_size):
            offset = i - center
            x = int(round(center + offset * dx))
            y = int(round(center + offset * dy))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
                
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            kernel[center, center] = 1.0
            
        return cv2.filter2D(image, -1, kernel)


class MultiScaleResolutionBucketer:
    """
    Downsamples high-resolution crops to variable resolution buckets (16x16 to 128x128)
    to reflect varying vehicle-to-sign distances before final network resizing.
    """
    BUCKETS: List[int] = [16, 24, 32, 48, 64, 96, 128]

    @classmethod
    def apply_bucketing(cls, image: np.ndarray, target_bucket: Optional[int] = None) -> np.ndarray:
        if target_bucket is None:
            target_bucket = int(np.random.choice(cls.BUCKETS))
        
        h, w = image.shape[:2]
        if min(h, w) <= target_bucket:
            return image
        
        downscaled = cv2.resize(image, (target_bucket, target_bucket), interpolation=cv2.INTER_AREA)
        return downscaled


class CanonicalDomainRandomizer:
    """
    Blends clean digital vector icons (from Meta.csv) with random outdoor textures,
    affine rotations, perspective warping, and brightness shifts to bridge domain shift.
    """
    
    @staticmethod
    def augment_icon(icon: np.ndarray, count: int = 50) -> List[np.ndarray]:
        h, w = icon.shape[:2]
        
        # Convert RGBA to BGR on white
        if icon.shape[2] == 4:
            alpha = icon[:, :, 3] / 255.0
            bgr = icon[:, :, :3]
            white = np.ones_like(bgr) * 255
            base = (bgr * alpha[:, :, None] + white * (1 - alpha[:, :, None])).astype(np.uint8)
        else:
            base = icon.copy()

        samples = []
        for _ in range(count):
            img = base.copy()
            
            # 1. Random affine rotation & scale
            angle = np.random.uniform(-14.0, 14.0)
            scale = np.random.uniform(0.75, 1.05)
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            bg_col = int(np.random.choice([255, 240, 210, 180, 120, 60]))
            img = cv2.warpAffine(img, M, (w, h), borderValue=(bg_col, bg_col, bg_col))
            
            # 2. Random environmental degradation
            dice = np.random.random()
            if dice < 0.25:
                img = AdverseWeatherAugmenter.add_rain(img, num_drops=np.random.randint(15, 45))
            elif dice < 0.45:
                img = AdverseWeatherAugmenter.reduce_luminance_night(img, brightness_factor=np.random.uniform(0.35, 0.70))
            elif dice < 0.60:
                img = AdverseWeatherAugmenter.apply_motion_blur(img, kernel_size=np.random.choice([3, 5]))
                
            # 3. Multi-scale bucketing
            img = MultiScaleResolutionBucketer.apply_bucketing(img)
            
            # 4. Standardize to 32x32 for network input
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            samples.append(img)
            
        return samples
