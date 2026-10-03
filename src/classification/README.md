# Traffic Sign Classification Module (`src/classification/`)

This directory houses the GTSRB sign recognition and classification subsystem maintained by **Member C (Classification Lead)** and **Member D (Integration Lead)**.

---

## 📄 File Index & Detailed Descriptions

### 1. `base.py`
* **Purpose**: Defines the abstract base class contract `BaseClassifier` that all classification models must fulfill.
* **Key Methods**:
  * `classify(self, crop: np.ndarray) -> ClassificationResult`: Classifies a single image patch into one of the 43 GTSRB classes or returns `class_id = -1` for background rejection.
  * `classify_batch(self, crops: List[np.ndarray]) -> List[ClassificationResult]`: Batch inference across multiple sign crops in a single video frame.
* **Interconnections**: Inherited by `PyTorchClassifier`, `ColorHeuristicClassifier`, `MockClassifier`, and `modules/C_classification/classifier.py`.

---

### 2. `model.py`
* **Purpose**: Production deep-learning classifier adapter supporting PyTorch `.pt`, TorchScript, and ONNX models trained on GTSRB.
* **Key Capabilities**:
  * **Image Preprocessing**: Resizes crops to $(32 \times 32)$, converts BGR to RGB, normalizes pixels to $[0.0, 1.0]$, and formats into PyTorch tensor `(1, 3, 32, 32)`.
  * **Out-of-Distribution / Negative Background Rejection**: Implements a strict confidence threshold (`min_confidence=0.65`). If maximum softmax probability is below the threshold, the crop is labeled as `Unrecognized / Background` with `class_id = -1`, preventing non-sign objects (faces, shirts, background scenery) from being assigned random sign classes.
  * **Top-K Probabilities**: Calculates top-5 most likely classes with human-readable names from `src.gtsrb_classes`.
* **Interconnections**: Used by `TrafficSignPipeline` in production mode and evaluated by `modules/C_classification/evaluate.py`.

---

### 3. `mock.py`
* **Purpose**: Mock and heuristic classifiers for rapid unit testing and zero-dependency execution.
* **Key Classes**:
  * `MockClassifier`: Returns deterministic sign predictions for CI/CD unit testing.
  * `ColorHeuristicClassifier`: Fast HSV color-rule heuristic (e.g. blue dominancy $\to$ Mandatory Turn Right, red dominancy $\to$ Stop / Speed Limit).
* **Interconnections**: Used in `tests/test_adapters.py` and `tests/test_pipeline.py`.
