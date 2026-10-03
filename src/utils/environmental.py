"""
Environmental Pre-Conditioning Engine for Adverse Driving Conditions.
Maintained by Member D (Integration & Pipeline Lead).

Provides adaptive illumination normalization, Contrast-Limited Adaptive
Histogram Equalization (CLAHE), and atmospheric dehazing for rain, fog,
night-time driving, and headlight glare.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class EnvironmentalConditioner:
    """
    Analyzes scene environmental telemetry and dynamically applies
    photon pre-conditioning before detector localization and classification.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.5,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        night_luminance_threshold: float = 85.0,
        glare_luminance_threshold: float = 215.0,
        dark_channel_patch_size: int = 15,
        omega_dehaze: float = 0.75
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.night_luminance_threshold = night_luminance_threshold
        self.glare_luminance_threshold = glare_luminance_threshold
        self.dark_channel_patch_size = dark_channel_patch_size
        self.omega_dehaze = omega_dehaze

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size
        )

    def analyze_scene(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extracts physical illumination and contrast metrics from an image frame.
        """
        if frame is None or frame.size == 0:
            return {
                "mean_luminance": 0.0,
                "rms_contrast": 0.0,
                "is_night": False,
                "is_glare": False,
                "haze_metric": 0.0
            }

        # Convert to LAB for perceptual luminance analysis
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]

        mean_lum = float(np.mean(l_chan))
        std_lum = float(np.std(l_chan))  # RMS contrast

        # Estimate haze via min channel (dark channel prior)
        min_chan = np.min(frame, axis=2)
        haze_metric = float(np.mean(min_chan))

        is_night = mean_lum < self.night_luminance_threshold
        is_glare = mean_lum > self.glare_luminance_threshold and std_lum > 65.0

        return {
            "mean_luminance": round(mean_lum, 2),
            "rms_contrast": round(std_lum, 2),
            "is_night": is_night,
            "is_glare": is_glare,
            "haze_metric": round(haze_metric, 2)
        }

    def apply_gamma(self, frame: np.ndarray, gamma: float = 1.4) -> np.ndarray:
        """
        Applies non-linear gamma correction via lookup table.
        gamma > 1.0 brightens shadows (night); gamma < 1.0 compresses highlights (glare).
        """
        inv_gamma = 1.0 / max(1e-4, gamma)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE on L-channel in LAB color space to preserve chromatic fidelity.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        merged = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def apply_dehaze(self, frame: np.ndarray) -> np.ndarray:
        """
        Fast atmospheric scattering reduction using dark channel prior.
        Particularly effective for fog, heavy rain mist, and highway spray.
        """
        norm_img = frame.astype(np.float32) / 255.0
        min_chan = np.min(norm_img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dark_channel_patch_size, self.dark_channel_patch_size))
        dark_channel = cv2.erode(min_chan, kernel)

        # Estimate atmospheric light A (top 0.1% brightest in dark channel)
        num_pixels = dark_channel.size
        num_top = max(1, int(num_pixels * 0.001))
        flat_dark = dark_channel.flatten()
        indices = np.argpartition(flat_dark, -num_top)[-num_top:]
        flat_img = norm_img.reshape(-1, 3)
        atmospheric_light = np.mean(flat_img[indices], axis=0)
        atmospheric_light = np.clip(atmospheric_light, 0.2, 1.0)

        # Transmission map estimation
        transmission = 1.0 - self.omega_dehaze * (dark_channel / np.max(atmospheric_light))
        transmission = np.clip(transmission, 0.15, 1.0)
        transmission = cv2.GaussianBlur(transmission, (15, 15), 0)

        # Recover scene radiance: J = (I - A) / max(t, 0.1) + A
        recovered = np.empty_like(norm_img)
        for c in range(3):
            recovered[:, :, c] = (norm_img[:, :, c] - atmospheric_light[c]) / transmission + atmospheric_light[c]

        recovered = np.clip(recovered * 255.0, 0, 255).astype(np.uint8)
        return recovered

    def auto_enhance(
        self,
        frame: np.ndarray,
        force_clahe: bool = False,
        force_dehaze: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Intelligently assesses frame conditions and executes optimal environmental pipeline.
        Returns:
            (enhanced_frame, telemetry_dict)
        """
        if frame is None or frame.size == 0:
            return frame, {"applied": "none"}

        telemetry = self.analyze_scene(frame)
        enhanced = frame.copy()
        applied_ops = []

        # 1. Night / Low-light compensation
        if telemetry["is_night"]:
            # Adaptive night gamma expansion (dynamically adjusts gamma = 1.35 - 1.85)
            lum = telemetry["mean_luminance"]
            gamma_val = round(float(np.clip(1.35 + 0.50 * max(0.0, 1.0 - lum / 45.0), 1.35, 1.85)), 2)
            enhanced = self.apply_gamma(enhanced, gamma=gamma_val)
            applied_ops.append(f"night_gamma_{gamma_val:.2f}")

        # 2. Glare reduction
        elif telemetry["is_glare"]:
            enhanced = self.apply_gamma(enhanced, gamma=0.82)
            applied_ops.append("glare_compression_0.82")

        # 3. Local contrast equalization (CLAHE)
        # Activated in low-light, high contrast, or when forced
        if telemetry["is_night"] or telemetry["rms_contrast"] < 38.0 or force_clahe:
            enhanced = self.apply_clahe(enhanced)
            applied_ops.append("clahe")

        # 4. Dehazing for fog/rain mist
        if (telemetry["haze_metric"] > 90.0 and not telemetry["is_glare"]) or force_dehaze:
            enhanced = self.apply_dehaze(enhanced)
            applied_ops.append("dehaze")

        telemetry["applied_ops"] = applied_ops
        telemetry["was_enhanced"] = len(applied_ops) > 0
        return enhanced, telemetry


_CONDITIONER_INSTANCE: Optional[EnvironmentalConditioner] = None


def get_environmental_conditioner() -> EnvironmentalConditioner:
    """Singleton getter for EnvironmentalConditioner."""
    global _CONDITIONER_INSTANCE
    if _CONDITIONER_INSTANCE is None:
        _CONDITIONER_INSTANCE = EnvironmentalConditioner()
    return _CONDITIONER_INSTANCE
