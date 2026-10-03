# Core Source Code Architecture (`src/`)

This directory contains the central integration pipeline, data contracts, and training logic designed by **Member D (Integration Lead)**.

---

## 📄 File Index & Detailed Descriptions

### 1. `schema.py`
* **Purpose**: Defines the universal Pydantic and Dataclass data contracts that all modules must adhere to.
* **Key Classes**:
  * `BoundingBox`: Represents sign coordinates `(x1, y1, x2, y2)`. Contains safety methods: `.clamp(max_w, max_h)` to prevent out-of-frame exceptions, `.pad(ratio)` to add context around sign crops, and area/aspect ratio calculations.
  * `DetectionResult`: Output of the detection stage (`bbox`, `confidence`, `detector_label`, `detector_class_id`).
  * `ClassificationResult`: Output of the classification stage (`class_id`, `class_name`, `confidence`, `category`, `top_k`).
  * `PipelineDetection`: Fused object combining a detection, its classification result, and its cropped image array.
  * `PipelineResult`: Full frame output including annotated BGR frame, list of `PipelineDetection` objects, latency breakdown dictionary (`detect_ms`, `classify_ms`, `track_ms`), FPS, active speed limit, and active warnings.
  * `SignCategory`: Enum categorizing signs into `PROHIBITORY`, `DANGER`, `MANDATORY`, and `OTHER`.
* **Interconnections**: Imported by virtually every file (`pipeline.py`, `tracker.py`, `visualizer.py`, `yolo.py`, `model.py`, `app.py`, `api.py`).

---

### 2. `gtsrb_classes.py`
* **Purpose**: Complete reference dictionary for all 43 classes of the German Traffic Sign Recognition Benchmark (GTSRB).
* **Key Components**:
  * `GTSRB_CLASSES`: Dict mapping class IDs `0` through `42` to official German traffic sign names (e.g. `0: "Speed limit (20km/h)"`, `14: "Stop"`, `22: "Bumpy road"`).
  * `GTSRB_CATEGORIES`: Categorizes each class ID into `SignCategory` for ADAS behavioral filtering.
  * `CATEGORY_COLORS`: Standardized BGR color palettes for HUD bounding boxes (Red for Prohibitory, Orange for Danger, Blue for Mandatory, Green for Priority/Other).
  * Helper functions: `get_class_name(cid)`, `get_sign_category(cid)`, `get_category_color(category)`.
* **Interconnections**: Used by `model.py`, `tracker.py`, `visualizer.py`, `benchmark_loader.py`, and `app.py`.

---

### 3. `pipeline.py`
* **Purpose**: Master orchestrator implementing the 4-stage Computer Vision pipeline:
  1. **Stage 1 (Detection)**: Passes frame to `self.detector.detect()`.
  2. **Stage 2 (Cropping & Clamping)**: Pads and extracts candidate image patches safely.
  3. **Stage 3 (Classification)**: Runs candidate crops through `self.classifier.classify_batch()`.
  4. **Stage 4 (Temporal Tracking & Telemetry)**: Updates `TemporalSignTracker` across video frames, tracks active speed limits, and renders HUD.
* **Key Class**: `TrafficSignPipeline` with factory `.create(mode='production' | 'heuristic' | 'mock')`.
* **Interconnections**: The central bridge connecting Member B's detector, Member C's classifier, Member A's preprocessing, the UI (`app.py`), and the REST API (`api.py`).

---

### 4. `train_custom.py`
* **Purpose**: PyTorch deep convolutional neural network definition and training engine for custom datasets or retraining on GTSRB.
* **Key Components**:
  * `TrafficSignCNN`: 3-block Convolutional Neural Network with BatchNorm, MaxPool2d, Dropout (0.25 and 0.50), and Fully Connected layers.
  * `train_model(...)`: Automated training loop featuring Adam optimizer, CrossEntropyLoss, validation accuracy evaluation, and TorchScript export (`torch.jit.trace`) for high-speed inference.
* **Interconnections**: Used by `app.py` for dynamic on-the-fly custom dataset training and by `modules/C_classification/train_classifier.py`.

---

### 5. `live_feed.py`
* **Purpose**: High-throughput OpenCV continuous video loop for standalone deployment on dashcams or robot cars without web browsers.
* **Interconnections**: Directly instantiates `TrafficSignPipeline` and displays output using standard `cv2.imshow()`.
