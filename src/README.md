# Core Perception Engine Architecture (`src/`)

This directory houses the central perception pipeline, universal data contracts, temporal tracking state machine, and model integration adapters.

---

## 📄 Subsystem Directory Map

| Directory / Module | Description |
| :--- | :--- |
| **`config.py`** | Master system configuration: paths, hardware acceleration, thresholds, and taxonomy mappings. |
| **`schema.py`** | Universal Pydantic models and dataclasses defining strict data contracts across all perception stages. |
| **`pipeline.py`** | Master orchestration pipeline connecting detection, cropping, classification, tracking, and HUD rendering. |
| **`gtsrb_classes.py`** | Complete 43-class GTSRB metadata, category enums, and color coding specifications. |
| **`data_pipeline/`** | Dataset ingestion, corrupt image cleaning, stratified train/val/test splitting, and orientation-preserving augmentation. |
| **`detection/`** | YOLOv8 neural detection engine, aspect-ratio clamping, and HSV geometric contour fallback. |
| **`classification/`** | Keras 3 and PyTorch CNN classification adapters, out-of-distribution filtering, and TT100K consensus engine. |
| **`tracking/`** | Temporal multi-object tracker, IoU exponential smoothing, and active ADAS speed limit state retention. |
| **`utils/`** | Cockpit HUD visualizer, CLAHE environmental conditioners, audio alert transducers, and streaming utilities. |
| **`dataset/`** | Benchmark ground-truth verification and custom dataset loaders. |

---

## ⚙️ Data Contract Architecture (`schema.py`)

All components communicate through immutable, validated dataclasses:

* **`BoundingBox`**: Coordinates `(x1, y1, x2, y2)` with `.clamp()`, `.pad()`, and `.iou()` methods.
* **`DetectionResult`**: Spatial localization output containing bounding box, confidence, and detector source.
* **`ClassificationResult`**: Fine-grained recognition output containing class ID, name, category, top-k probabilities, and speed limit metadata.
* **`PipelineDetection`**: Fused object binding localized boxes to classification posteriors and cropped image arrays.
* **`PipelineResult`**: Final frame result containing annotated visual frames, active speed limits, hazard banners, and latency breakdowns.
