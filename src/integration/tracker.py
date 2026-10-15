"""Temporal Sign Tracking and Speed Limit Memory (Member D).
Maintains sign track continuity across frames and remembers active speed limits.
"""

from typing import List, Optional, Dict
import numpy as np
from src.schema import BoundingBox, IntegratedDetection


class SignTrack:
    def __init__(self, track_id: int, detection: IntegratedDetection, frame_idx: int):
        self.track_id = track_id
        self.bbox = detection.detection.bbox
        self.class_id = detection.final_class_id
        self.class_name = detection.final_class_name
        self.confidence = detection.final_confidence
        self.first_seen = frame_idx
        self.last_seen = frame_idx
        self.hits = 1
        self.speed_limit = (
            detection.classification.speed_limit
            if detection.classification
            else None
        )

    def update(self, detection: IntegratedDetection, frame_idx: int):
        self.bbox = detection.detection.bbox
        # Exponential moving average for confidence
        self.confidence = 0.7 * detection.final_confidence + 0.3 * self.confidence
        self.last_seen = frame_idx
        self.hits += 1
        if detection.classification and detection.classification.speed_limit is not None:
            self.speed_limit = detection.classification.speed_limit


def calculate_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxAArea = boxA.area
    boxBArea = boxB.area
    union_area = float(boxAArea + boxBArea - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


class SignTracker:
    def __init__(self, max_missed_frames: int = 15, iou_match_thresh: float = 0.35):
        self.max_missed_frames = max_missed_frames
        self.iou_match_thresh = iou_match_thresh
        self.next_track_id = 1
        self.tracks: Dict[int, SignTrack] = {}
        self.current_speed_limit: Optional[int] = None

    def update(self, detections: List[IntegratedDetection], frame_idx: int) -> List[IntegratedDetection]:
        """Match incoming detections with active tracks, assign track_ids, and update speed memory."""
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        # Greedy IoU matching
        for t_id, track in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = -1

            for d_idx in unmatched_dets:
                det = detections[d_idx]
                iou = calculate_iou(track.bbox, det.detection.bbox)
                if iou > best_iou and iou >= self.iou_match_thresh:
                    best_iou = iou
                    best_det_idx = d_idx

            if best_det_idx != -1:
                det = detections[best_det_idx]
                track.update(det, frame_idx)
                det.track_id = t_id
                matched_tracks.add(t_id)
                unmatched_dets.remove(best_det_idx)

                # Update speed limit memory
                if track.speed_limit is not None:
                    self.current_speed_limit = track.speed_limit
                elif track.class_id in [6, 32]:  # End of speed limit or all restrictions
                    self.current_speed_limit = None

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_track = SignTrack(self.next_track_id, det, frame_idx)
            det.track_id = self.next_track_id
            self.tracks[self.next_track_id] = new_track

            if new_track.speed_limit is not None:
                self.current_speed_limit = new_track.speed_limit
            elif new_track.class_id in [6, 32]:
                self.current_speed_limit = None

            self.next_track_id += 1

        # Prune expired tracks
        expired = [
            t_id for t_id, track in self.tracks.items()
            if (frame_idx - track.last_seen) > self.max_missed_frames
        ]
        for t_id in expired:
            del self.tracks[t_id]

        return detections
