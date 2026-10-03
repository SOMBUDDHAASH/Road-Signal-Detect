"""
Temporal Sign Tracker & ADAS State Machine for Continuous Driving Streams.
Prevents frame-by-frame flickering, smooths bounding boxes, and tracks active vehicle speed limits.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.schema import PipelineDetection, BoundingBox, ClassificationResult, DetectionResult, SignCategory
from src.gtsrb_classes import get_class_name, get_sign_category


def compute_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = boxA.area
    boxB_area = boxB.area
    union_area = float(boxA_area + boxB_area - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class TrackedSign:
    """Represents a continuous track of a detected traffic sign across frames."""
    track_id: int
    bbox: BoundingBox
    classification: ClassificationResult
    detection_conf: float
    crop: Optional[np.ndarray] = None
    hits: int = 1
    disappeared_count: int = 0
    history_classes: List[int] = field(default_factory=list)
    history_boxes: List[BoundingBox] = field(default_factory=list)

    def update(self, new_detection: PipelineDetection, smoothing_alpha: float = 0.65):
        """Update track with new frame detection and smooth coordinates."""
        self.hits += 1
        self.disappeared_count = 0
        self.detection_conf = new_detection.detection.confidence
        self.crop = new_detection.crop

        # Smooth bounding box
        nb = new_detection.detection.bbox
        ob = self.bbox
        smoothed_x1 = int(round(smoothing_alpha * nb.x1 + (1 - smoothing_alpha) * ob.x1))
        smoothed_y1 = int(round(smoothing_alpha * nb.y1 + (1 - smoothing_alpha) * ob.y1))
        smoothed_x2 = int(round(smoothing_alpha * nb.x2 + (1 - smoothing_alpha) * ob.x2))
        smoothed_y2 = int(round(smoothing_alpha * nb.y2 + (1 - smoothing_alpha) * ob.y2))
        self.bbox = BoundingBox(smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2)

        # Update classification history for majority voting (last 5 frames)
        cid = new_detection.classification.class_id
        self.history_classes.append(cid)
        if len(self.history_classes) > 5:
            self.history_classes.pop(0)

        # Stable class from mode with tie-breaking favoring most recent
        from collections import Counter
        counts = Counter(self.history_classes)
        max_freq = max(counts.values())
        candidates = [c for c, cnt in counts.items() if cnt == max_freq]
        majority_id = max(candidates, key=lambda c: len(self.history_classes) - 1 - self.history_classes[::-1].index(c))

        self.classification = ClassificationResult(
            class_id=majority_id,
            class_name=get_class_name(majority_id),
            confidence=max(self.classification.confidence, new_detection.classification.confidence),
            category=get_sign_category(majority_id),
            top_k=new_detection.classification.top_k
        )


class TemporalSignTracker:
    """
    Multi-sign tracker for continuous video streams.
    Smooths jitter, maintains track IDs, and tracks vehicle driving context (e.g. current speed limit).
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_disappeared: int = 6,
        min_hits_to_confirm: int = 2,
        smoothing_alpha: float = 0.65
    ):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.min_hits_to_confirm = min_hits_to_confirm
        self.smoothing_alpha = smoothing_alpha

        self.next_track_id = 1
        self.tracks: Dict[int, TrackedSign] = {}

        # ADAS Dashboard States
        self.current_speed_limit: Optional[str] = None
        self.active_hazard_warning: Optional[str] = None

    def update(self, detections: List[PipelineDetection]) -> List[PipelineDetection]:
        """
        Updates tracks with the latest frame's detections.
        Returns stabilized, smoothed PipelineDetection objects.
        """
        if not self.tracks and not detections:
            return []

        # If no active tracks, register all detections
        if not self.tracks:
            for d in detections:
                self._register_track(d)
            return self._get_active_pipeline_detections()

        # If no detections in current frame, age all tracks
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].disappeared_count += 1
                if self.tracks[track_id].disappeared_count > self.max_disappeared:
                    del self.tracks[track_id]
            return self._get_active_pipeline_detections()

        # Match existing tracks with detections via IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = compute_iou(self.tracks[tid].bbox, det.detection.bbox)

        matched_tracks = set()
        matched_detections = set()

        if iou_matrix.size > 0:
            # Greedy matching by highest IoU
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                row, col = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[row]
                self.tracks[tid].update(detections[col], self.smoothing_alpha)
                self._update_vehicle_state(self.tracks[tid])

                matched_tracks.add(tid)
                matched_detections.add(col)
                # Invalidate row and column
                iou_matrix[row, :] = -1.0
                iou_matrix[:, col] = -1.0

        # Unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].disappeared_count += 1
                if self.tracks[tid].disappeared_count > self.max_disappeared:
                    del self.tracks[tid]

        # Unmatched detections -> create new tracks
        for col, det in enumerate(detections):
            if col not in matched_detections:
                new_tid = self._register_track(det)
                self._update_vehicle_state(self.tracks[new_tid])

        return self._get_active_pipeline_detections()

    def _register_track(self, det: PipelineDetection) -> int:
        tid = self.next_track_id
        self.next_track_id += 1
        track = TrackedSign(
            track_id=tid,
            bbox=det.detection.bbox,
            classification=det.classification,
            detection_conf=det.detection.confidence,
            crop=det.crop,
            history_classes=[det.classification.class_id]
        )
        self.tracks[tid] = track
        return tid

    def _update_vehicle_state(self, track: TrackedSign):
        # Speed limits (Classes 0-5, 7, 8 and MPH advisory)
        cid = track.classification.class_id
        cname = track.classification.class_name
        speed_map = {
            0: "20 km/h", 1: "30 km/h", 2: "50 km/h", 3: "60 km/h",
            4: "70 km/h", 5: "80 km/h", 7: "100 km/h", 8: "120 km/h"
        }
        if ("M.P.H." in cname or "MPH" in cname) and track.hits >= self.min_hits_to_confirm:
            # Extract speed string e.g. "20 M.P.H."
            if "(" in cname and ")" in cname:
                self.current_speed_limit = cname.split("(")[-1].split(")")[0]
            else:
                self.current_speed_limit = cname
        elif cid in speed_map and track.hits >= self.min_hits_to_confirm:
            self.current_speed_limit = speed_map[cid]
        elif cid in (6, 32) and track.hits >= self.min_hits_to_confirm:
            # End of speed limits
            self.current_speed_limit = "No Limit"

        # Hazards (Danger class)
        if track.classification.category == SignCategory.DANGER and track.hits >= self.min_hits_to_confirm:
            self.active_hazard_warning = track.classification.class_name
        elif track.disappeared_count > 2 and self.active_hazard_warning == track.classification.class_name:
            self.active_hazard_warning = None

    def _get_active_pipeline_detections(self) -> List[PipelineDetection]:
        """Convert confirmed tracks back to PipelineDetection list."""
        active_items: List[PipelineDetection] = []
        for tid, trk in self.tracks.items():
            # Only display if track has persisted at least min_hits or is fresh
            if trk.hits >= self.min_hits_to_confirm or trk.disappeared_count == 0:
                active_items.append(PipelineDetection(
                    detection=DetectionResult(
                        bbox=trk.bbox,
                        confidence=trk.detection_conf,
                        detector_label=f"Track #{trk.track_id}",
                        detector_class_id=0
                    ),
                    classification=trk.classification,
                    crop=trk.crop
                ))
        return active_items
