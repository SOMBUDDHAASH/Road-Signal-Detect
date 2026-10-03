# Temporal Tracking & ADAS State Module (`src/tracking/`)

This directory houses the temporal tracking, anti-flicker smoothing, and Advanced Driver Assistance Systems (ADAS) state machine designed by **Member D (Integration Lead)**.

---

## 📄 File Index & Detailed Descriptions

### 1. `tracker.py`
* **Purpose**: Solves the fundamental problem of frame-by-frame object detection in continuous driving: detections flicker, bounding boxes jitter, and classifications fluctuate due to motion blur and lighting variations.
* **Key Components**:
  * `TrackedSign`: Dataclass tracking an individual physical sign over time (`track_id`, `bbox`, `history_bboxes`, `class_history`, `hits`, `time_since_update`).
  * `compute_iou(boxA, boxB)`: Calculates Intersection over Union (IoU) between bounding boxes across sequential video frames.
  * `TemporalSignTracker`: Master tracker class:
    * **IoU Matching**: Associates new detections with existing tracks using a greedy matching algorithm (IoU $\ge 0.30$).
    * **Exponential Coordinate Smoothing**: Smooths bounding box coordinates across frames to eliminate jitter:
      $$\text{bbox}_{\text{smooth}} = \alpha \cdot \text{bbox}_{\text{new}} + (1 - \alpha) \cdot \text{bbox}_{\text{old}}$$
    * **Temporal Majority Voting**: Accumulates classification predictions across a rolling window of frames and takes the statistical majority vote, preventing single-frame misclassifications from triggering false alarms.
    * **ADAS State Machine**:
      - **Active Speed Limit Memory**: Remembers the latest confirmed speed limit sign (e.g. 50 km/h) and keeps it active on the vehicle dashboard until an "End of speed limits" sign (Classes 6 or 32) is encountered or the track expires.
      - **Active Hazard Warnings**: Triggers prominent cockpit alert banners when approaching danger signs (e.g. Bumpy Road, Slippery Road, Road Work, Pedestrians).
* **Interconnections**: Integrated directly inside `TrafficSignPipeline.process_frame()` and rendered onto the cockpit HUD by `Visualizer`.
