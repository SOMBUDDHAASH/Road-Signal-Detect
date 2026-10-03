"""
Base classifier contract. Member C must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.schema import ClassificationResult


class BaseClassifier(ABC):
    """
    Abstract interface for GTSRB traffic sign classification (43 classes).
    Member C (Classification Lead) must adhere to this interface.
    """

    @abstractmethod
    def classify(self, crop: np.ndarray) -> ClassificationResult:
        """
        Classifies a single cropped image patch into one of 43 GTSRB classes.

        Parameters:
            crop (np.ndarray): Cropped sign patch in BGR or RGB format (H, W, C).

        Returns:
            ClassificationResult: Class ID (0-42), human name, confidence, category, and top-k.
        """
        pass

    def classify_batch(self, crops: List[np.ndarray]) -> List[ClassificationResult]:
        """
        Batch classification for multiple crops in a single frame.
        Default implementation loops through classify(), but subclasses can override for tensor batching.
        """
        return [self.classify(c) for c in crops]
