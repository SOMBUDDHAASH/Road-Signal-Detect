# Temporal Tracking & ADAS State Machine (`src/tracking/`)

This directory houses the temporal tracking, anti-flicker coordinate smoothing, and Advanced Driver Assistance Systems (ADAS) vehicle state engine.

---

## 📌 Architecture & Responsibilities

In real-world continuous driving video, single-frame object detection often suffers from bounding box jitter and momentary detection drops caused by motion blur, occlusions, or lighting changes. The temporal tracking module solves these challenges:

1. **Multi-Object Association**: Links sign detections across sequential frames using Intersection over Union (IoU $\ge 0.45$).
2. **Exponential Coordinate Smoothing**:
   $$\text{bbox}_{\text{smooth}} = \alpha \cdot \text{bbox}_{\text{new}} + (1 - \alpha) \cdot \text{bbox}_{\text{old}}$$
   Smooths jittery bounding boxes to produce stable automotive HUD overlays.
3. **Temporal Majority Voting**: Accumulates classification predictions across a rolling window of frames, preventing isolated single-frame misclassifications from triggering false alerts.
4. **Active Speed Limit Memory**: Stores the active speed limit in vehicle memory, holding the speed restriction until explicitly replaced by a new limit or an "End of speed limit" sign (GTSRB Class 32).
5. **Hazard Alert State**: Triggers persistent cockpit alert banners when approaching danger zones (e.g. Bumpy Road, Slippery Road, Pedestrian Crossing).
