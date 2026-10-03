"""Classification module interfaces and implementations."""

from src.classification.base import BaseClassifier
from src.classification.mock import MockClassifier, ColorHeuristicClassifier

__all__ = ["BaseClassifier", "MockClassifier", "ColorHeuristicClassifier"]
