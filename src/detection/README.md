# Traffic Sign Detection Module (`src/detection/`)

This directory houses the traffic sign localization subsystem maintained by **Member B (Detection Lead)** and **Member D (Integration Lead)**.

---

## 📄 File Index & Detailed Descriptions

### 1. `base.py`
* **Purpose**: Defines the abstract base class contract `BaseDetector` that all detector implementations must fulfill.
* **Key Method**:
  ```python
  @abstractmethod
  def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
      """Takes an image frame (H, W, 3) and returns localized bounding boxes with confidence."""
  ```
* **Interconnections**: Inherited by `YOLODetector`, `RobustTrafficSignDetector`, `MockDetector`, and `modules/B_detection/detector.py`.

---

### 2. `yolo.py`
* **Purpose**: Production deep-learning adapter that executes YOLOv8/v11 object detection on incoming video frames using Ultralytics.
* **Key Capabilities**:
  * **Fine-Tuned Weight Loading**: Loads `weights/detection/best.pt`.
  * **Anti-Person / Anti-Body Class Filtering**: Contains a strict filter rejecting non-traffic-sign classes (e.g. `person`, `human`, `face`, `body`, `chair`, `couch`, `laptop`, `cell phone`), preventing people sitting in front of webcams from being treated as road signs.
  * **Aspect Ratio & Oversize Guard**: Rejects degenerate boxes ($<12$ px) or wide horizontal boxes spanning $>70\%$ of the screen.
  * **Automatic Fallback**: If weights are missing, falls back cleanly to `RobustTrafficSignDetector`.
* **Interconnections**: Instantiated by `TrafficSignPipeline` in production mode and by `app.py`.

---

### 3. `shape_detector.py`
* **Purpose**: Robust geometric and computer vision shape detector operating without deep learning weights.
* **Key Capabilities**:
  * **Geometric Analysis**: Detects circular signs (Hough Circles & high circularity contours) and triangular/octagonal warning signs (`approxPolyDP` polygon vertex matching).
  * **Contrast Verification**: Uses `has_sign_symbol_contrast()` (Canny edge density $\ge 0.035$) to ensure the crop contains an internal road symbol, rejecting plain walls or solid shirts.
  * **Skin Tone Exclusion**: Computes YCrCb skin chrominance ratios to reject hands or faces holding signs.
* **Interconnections**: Used as the primary fallback in `YOLODetector` and available in `app.py` under heuristic mode.

---

### 4. `mock.py`
* **Purpose**: Lightweight detectors for unit tests and instant local demonstrations without GPU or heavy dependencies.
* **Key Classes**:
  * `MockDetector`: Returns synthetic predetermined bounding boxes for reproducible unit testing.
  * `ColorContourDetector`: Fast HSV color thresholding (Red/Blue/Yellow) for basic geometric testing.
* **Interconnections**: Used extensively in `tests/test_adapters.py` and `tests/test_pipeline.py`.
