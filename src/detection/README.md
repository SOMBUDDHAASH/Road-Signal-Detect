# Traffic Sign Detection Architecture (`src/detection/`)

This directory houses the real-time traffic sign localization subsystem, integrating YOLOv8 deep learning detection with geometric and HSV color-space fallback mechanisms.

---

## 📄 File Index & Specifications

### 1. `base.py`
* **Purpose**: Defines the abstract base class contract `BaseDetector` that all detector implementations must fulfill:
  ```python
  @abstractmethod
  def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
      """Takes an image frame (H, W, 3) and returns localized bounding boxes with confidence."""
  ```

---

### 2. `yolo.py` & `yolo_detector.py`
* **Purpose**: Production deep-learning adapters executing YOLOv8 inference using Ultralytics.
* **Key Capabilities**:
  * **Fine-Tuned Weight Loading**: Loads `modules/B_detection/models/best.pt` or `weights/detection/best.pt`.
  * **Anti-Person / Anti-Body Class Filtering**: Strict class filters reject non-traffic-sign objects (pedestrians, faces, chairs, laptops), preventing people in front of webcams from generating false positives.
  * **Aspect Ratio & Oversize Guard**: Rejects degenerate boxes ($<12$ px) or abnormal aspect ratios.

---

### 3. `shape_detector.py`
* **Purpose**: Heuristic geometric and color contour detector operating without deep learning dependencies.
* **Key Capabilities**:
  * **Dual-Hue Red Masking**: Captures both lower and upper red bands in HSV space.
  * **Morphological Regularization**: Elliptical kernel smoothing and contour area filtering.
  * **Circularity & Extent Validation**: Matches geometric properties of standard circular, triangular, and octagonal signs.

---

### 4. `preprocess.py`
* **Purpose**: Multi-scale letterboxing ($640 \times 640$) and CLAHE contrast enhancement for low-light road conditions.
