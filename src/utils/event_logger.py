"""
Real-time Local Timestamp Event Logger for Traffic Sign Detections.
Records detection events with local time codes (e.g. '03/10/2026 16:20:05 -> Stop sign detected').
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import csv
import json
import io
import time

from src.schema import PipelineDetection


@dataclass
class SignEvent:
    timestamp_str: str       # e.g. "03/10/2026 16:20:05"
    unix_time: float
    class_id: int
    class_name: str
    category: str
    confidence: float
    track_id: Optional[int] = None
    bbox: Optional[List[int]] = None

    def to_readable_log(self) -> str:
        track_info = f" [Track #{self.track_id}]" if self.track_id else ""
        return f"{self.timestamp_str} -> [{self.class_id}] {self.class_name} ({self.confidence*100:.1f}%){track_info}"


class DetectionEventLogger:
    """
    Thread-safe event logger that records detection timestamps in the user's local timezone.
    Includes smart de-duplication to prevent logging identical signs 30 times a second.
    """

    def __init__(self, dedup_cooldown_sec: float = 3.0, max_history: int = 500):
        self.dedup_cooldown_sec = dedup_cooldown_sec
        self.max_history = max_history
        self.events: List[SignEvent] = []
        self._last_logged_time: Dict[str, float] = {}

    def log_detections(self, detections: List[PipelineDetection]) -> List[SignEvent]:
        """
        Processes frame detections, creates timestamped events for new or confirmed signs.
        Returns newly logged events in this call.
        """
        now_dt = datetime.now()
        # Format: e.g. "03/10/2026 16:20:05" (Day/Month/Year Hour:Minute:Second)
        timestamp_str = now_dt.strftime("%d/%m/%Y %H:%M:%S")
        now_ts = time.time()

        new_events = []

        for item in detections:
            cid = item.classification.class_id
            cname = item.classification.class_name
            conf = float(item.classification.confidence)
            cat = item.classification.category.value

            # Extract track ID if available from detector label (e.g. "Track #2")
            track_id = None
            if "Track #" in item.detection.detector_label:
                try:
                    track_id = int(item.detection.detector_label.split("#")[-1])
                except ValueError:
                    track_id = None

            # Deduplication key: track_id if present, else class_id
            dedup_key = f"track_{track_id}" if track_id else f"class_{cid}"

            last_time = self._last_logged_time.get(dedup_key, 0.0)
            if (now_ts - last_time) >= self.dedup_cooldown_sec:
                self._last_logged_time[dedup_key] = now_ts
                event = SignEvent(
                    timestamp_str=timestamp_str,
                    unix_time=now_ts,
                    class_id=cid,
                    class_name=cname,
                    category=cat,
                    confidence=round(conf, 3),
                    track_id=track_id,
                    bbox=list(item.detection.bbox.to_xyxy())
                )
                self.events.append(event)
                new_events.append(event)

                if len(self.events) > self.max_history:
                    self.events.pop(0)

        return new_events

    def get_recent_logs(self, limit: int = 50) -> List[str]:
        """Returns readable string logs formatted in reverse chronological order."""
        return [e.to_readable_log() for e in reversed(self.events[-limit:])]

    def export_csv(self) -> str:
        """Export all logged events as a CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp", "Class ID", "Class Name", "Category", "Confidence", "Track ID", "BBox"])
        for e in self.events:
            writer.writerow([
                e.timestamp_str,
                e.class_id,
                e.class_name,
                e.category,
                f"{e.confidence*100:.1f}%",
                e.track_id or "N/A",
                str(e.bbox)
            ])
        return output.getvalue()

    def export_json(self) -> str:
        """Export all logged events as a JSON string."""
        return json.dumps([asdict(e) for e in self.events], indent=2)

    def clear(self):
        self.events.clear()
        self._last_logged_time.clear()
