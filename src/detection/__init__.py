"""Detection module interfaces and implementations."""

from src.detection.base import BaseDetector
from src.detection.mock import MockDetector, ColorContourDetector

__all__ = ["BaseDetector", "MockDetector", "ColorContourDetector"]
