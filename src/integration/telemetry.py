"""Real-time Telemetry & Event Logger (Member D).
Maintains structured event history for audit logs, telemetry graphs, and driver reports.
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path


class TelemetryLogger:
    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.events: List[Dict[str, Any]] = []

    def log_detection(
        self,
        frame_idx: int,
        track_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        speed_limit: int = None,
    ):
        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "frame_index": frame_idx,
            "track_id": track_id,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "speed_limit": speed_limit,
        }
        self.events.append(event)
        if len(self.events) > self.max_records:
            self.events.pop(0)

    def get_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        return self.events[-n:]

    def export_csv(self, file_path: str) -> str:
        out_p = Path(file_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if not self.events:
            return ""

        keys = ["timestamp", "frame_index", "track_id", "class_id", "class_name", "confidence", "speed_limit"]
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.events)
        return str(out_p)

    def export_json(self, file_path: str) -> str:
        out_p = Path(file_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2)
        return str(out_p)

    def clear(self):
        self.events.clear()
