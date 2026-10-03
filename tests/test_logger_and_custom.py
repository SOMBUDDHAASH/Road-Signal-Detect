"""
Unit tests for DetectionEventLogger and CustomDatasetManager.
"""

import os
import json
import pytest
from datetime import datetime

from src.utils.event_logger import DetectionEventLogger, SignEvent
from src.schema import PipelineDetection, DetectionResult, ClassificationResult, BoundingBox, SignCategory
from src.dataset.custom_dataset import CustomDatasetManager


def test_event_logger_timestamp_format():
    logger = DetectionEventLogger(dedup_cooldown_sec=1.0)
    det = PipelineDetection(
        detection=DetectionResult(bbox=BoundingBox(10, 10, 50, 50), confidence=0.95),
        classification=ClassificationResult(class_id=14, class_name="Stop", confidence=0.95, category=SignCategory.PROHIBITORY)
    )

    events = logger.log_detections([det])
    assert len(events) == 1
    ev = events[0]
    # Check timestamp format: DD/MM/YYYY HH:MM:SS
    assert "/" in ev.timestamp_str and ":" in ev.timestamp_str
    # Verify readable log output contains "Stop"
    log_text = ev.to_readable_log()
    assert "Stop" in log_text
    assert "95.0%" in log_text

    # CSV and JSON export
    csv_str = logger.export_csv()
    assert "Timestamp,Class ID,Class Name" in csv_str
    assert "Stop" in csv_str

    json_str = logger.export_json()
    assert "Stop" in json_str


def test_custom_dataset_manager(tmp_path):
    mgr = CustomDatasetManager(storage_dir=str(tmp_path / "custom"))
    dataset_dir = tmp_path / "test_data"
    os.makedirs(dataset_dir / "speed_40", exist_ok=True)
    os.makedirs(dataset_dir / "stop_sign", exist_ok=True)

    class_map = mgr.index_dataset(str(dataset_dir))
    assert len(class_map) == 2
    assert "Speed 40" in class_map.values()
    assert "Stop Sign" in class_map.values()
