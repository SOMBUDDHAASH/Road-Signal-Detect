"""
Base detector contract. Member B must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.schema import DetectionResult


class BaseDetector(ABC):
    """
    Abstract interface for traffic sign detection.
    Member B (Detection Lead) must adhere to this interface.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
        """
        Detect traffic signs within an input image frame.

        Parameters:
            image (np.ndarray): Input frame in BGR or RGB format (H, W, C).
            conf_threshold (float): Minimum confidence threshold (0.0 to 1.0) to keep a box.

        Returns:
            List[DetectionResult]: List of detected signs with bounding boxes and confidence.
        """
        pass
