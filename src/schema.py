"""Data contracts and schema definitions for the Traffic Sign Detection & Recognition pipeline.
Defines explicit function inputs, outputs, bounding box structures, and pipeline results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from pydantic import BaseModel, Field


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

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1.0, float(self.height))

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

    def pad(self, padding_ratio: Union[float, int] = 0.0, max_width: int = 10000, max_height: int = 10000) -> BoundingBox:
        """Expands the box outward by a fractional margin or fixed pixel padding."""
        if isinstance(padding_ratio, float) and padding_ratio < 1.0:
            pad_x = int(self.width * padding_ratio)
            pad_y = int(self.height * padding_ratio)
        else:
            pad_x = int(padding_ratio)
            pad_y = int(padding_ratio)
        return BoundingBox(
            max(0, self.x1 - pad_x),
            max(0, self.y1 - pad_y),
            min(max_width, self.x2 + pad_x),
            min(max_height, self.y2 + pad_y)
        )

    def crop(self, image: np.ndarray, padding: Union[float, int] = 0) -> Optional[np.ndarray]:
        """Crop this region from an image array with optional padding."""
        h, w = image.shape[:2]
        box = self.pad(padding, w, h)
        if box.width <= 0 or box.height <= 0:
            return None
        return image[box.y1 : box.y2, box.x1 : box.x2].copy()

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
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    detector_source: str = "YOLOv8"

    def __post_init__(self):
        if self.class_id is None:
            self.class_id = self.detector_class_id
        if self.class_name is None:
            self.class_name = self.detector_label


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
    speed_limit: Optional[int] = None


@dataclass
class TT100KResult:
    """Contract for Secondary TT100K (Tsinghua-Tencent 100K) Analysis Model."""
    class_code: str
    class_name: str
    confidence: float
    mapped_gtsrb_id: Optional[int] = None
    is_consensus: bool = False
    consensus_note: str = ""
    top_k: List[Tuple[str, str, float]] = field(default_factory=list)


@dataclass
class PipelineDetection:
    """Combined detection + classification result with extracted cropped patch."""
    detection: DetectionResult
    classification: ClassificationResult
    crop: Optional[np.ndarray] = None
    tt100k_result: Optional[TT100KResult] = None


@dataclass
class IntegratedDetection:
    """Integrated detection item compatible with Member D pipeline."""
    detection: DetectionResult
    classification: Optional[ClassificationResult] = None
    final_class_id: int = 0
    final_class_name: str = ""
    final_confidence: float = 0.0
    track_id: Optional[int] = None


@dataclass
class PipelineFrameResult:
    """Per-frame result container for video and stream perception."""
    frame_index: int
    original_shape: Tuple[int, int, int]
    detections: List[IntegratedDetection]
    active_speed_limit: Optional[int] = None
    inference_time_ms: float = 0.0
    annotated_frame: Optional[np.ndarray] = None


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


# Pydantic schemas for REST API serialization
class BBoxModel(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int


class DetectionItemModel(BaseModel):
    bbox: BBoxModel
    confidence: float
    class_id: int
    class_name: str
    category: str
    speed_limit: Optional[int] = None


class PipelineResponseModel(BaseModel):
    status: str
    image_width: int
    image_height: int
    total_detections: int
    active_speed_limit: Optional[int]
    inference_time_ms: float
    detections: List[DetectionItemModel]
