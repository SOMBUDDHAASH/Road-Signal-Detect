"""
Data contracts and schema definitions for the Traffic Sign Detection & Recognition pipeline.
Defines explicit function inputs, outputs, bounding box structures, and pipeline results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class SignCategory(str, Enum):
    PROHIBITORY = "Prohibitory"       # Red circle (speed limit, no entry, etc.)
    DANGER = "Danger"                 # Red triangle (curves, pedestrians, road work)
    MANDATORY = "Mandatory"           # Blue circle (turn left/right, roundabout)
    OTHER = "Other"                   # Priority road, yield, derestriction, etc.


@dataclass
class BoundingBox:
    """Bounding box coordinates in pixels: (x1, y1) top-left, (x2, y2) bottom-right."""
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        # Ensure integers
        self.x1 = int(round(self.x1))
        self.y1 = int(round(self.y1))
        self.x2 = int(round(self.x2))
        self.y2 = int(round(self.y2))
        # Ensure x1 <= x2 and y1 <= y2
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.width, self.height)

    def clamp(self, max_width: int, max_height: int) -> BoundingBox:
        """Clamp coordinates within image boundaries."""
        cx1 = max(0, min(self.x1, max_width))
        cy1 = max(0, min(self.y1, max_height))
        cx2 = max(0, min(self.x2, max_width))
        cy2 = max(0, min(self.y2, max_height))
        return BoundingBox(cx1, cy1, cx2, cy2)

    def pad(self, padding_ratio: float, max_width: int, max_height: int) -> BoundingBox:
        """Expands the box outward by a fractional margin of width/height."""
        pad_x = int(self.width * padding_ratio)
        pad_y = int(self.height * padding_ratio)
        return BoundingBox(
            max(0, self.x1 - pad_x),
            max(0, self.y1 - pad_y),
            min(max_width, self.x2 + pad_x),
            min(max_height, self.y2 + pad_y)
        )

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union (IoU) with another bounding box."""
        xA = max(self.x1, other.x1)
        yA = max(self.y1, other.y1)
        xB = min(self.x2, other.x2)
        yB = min(self.y2, other.y2)
        inter = max(0, xB - xA) * max(0, yB - yA)
        union = self.area + other.area - inter
        return float(inter / union) if union > 0 else 0.0


@dataclass
class DetectionResult:
    """Contract for Member B (Detection Lead). Output of detect()."""
    bbox: BoundingBox
    confidence: float
    detector_label: str = "traffic_sign"
    detector_class_id: int = 0


@dataclass
class ClassificationResult:
    """Contract for Member C (Classification Lead). Output of classify()."""
    class_id: int
    class_name: str
    confidence: float
    category: SignCategory = SignCategory.OTHER
    top_k: List[Tuple[int, str, float]] = field(default_factory=list)
    entropy: float = 0.0
    margin: float = 1.0
    is_ambiguous: bool = False


@dataclass
class PipelineDetection:
    """Combined detection + classification result with extracted cropped patch."""
    detection: DetectionResult
    classification: ClassificationResult
    crop: Optional[np.ndarray] = None


@dataclass
class PipelineResult:
    """Final output of the end-to-end integration pipeline."""
    original_frame: np.ndarray
    annotated_frame: np.ndarray
    detections: List[PipelineDetection]
    latency_ms: Dict[str, float]
    fps: float
    active_speed_limit: Optional[str] = None
    active_hazard: Optional[str] = None
    environmental_telemetry: Optional[Dict[str, Any]] = None

    @property
    def total_latency_ms(self) -> float:
        return self.latency_ms.get("total_ms", 0.0)

    @property
    def num_signs_detected(self) -> int:
        return len(self.detections)

