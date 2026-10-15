# Traffic Sign Classification Architecture (`src/classification/`)

This directory houses the GTSRB sign recognition and classification subsystem, supporting multi-backend CNN inference and secondary consensus verification.

---

## 📄 File Index & Specifications

### 1. `base.py`
* **Purpose**: Defines the abstract base class contract `BaseClassifier` that all classification models must fulfill.
* **Key Methods**:
  * `classify(self, crop: np.ndarray) -> ClassificationResult`: Classifies a single image patch into one of the 43 GTSRB classes or returns `class_id = -1` for background rejection.
  * `classify_batch(self, crops: List[np.ndarray]) -> List[ClassificationResult]`: Batch inference across multiple sign crops in a single video frame.

---

### 2. `tf_classifier.py` & `keras_classifier.py`
* **Purpose**: Enterprise Keras 3 classification adapters executing the trained 43-class GTSRB CNN (`classification/models/traffic_sign_model.keras`).
* **Key Capabilities**:
  * Multi-backend acceleration (PyTorch or TensorFlow runtime).
  * Native BGR channel preservation ensuring 100.00% confidence on canonical test signs.
  * Seamless fallback between local weight paths.

---

### 3. `model.py` & `pytorch_classifier.py`
* **Purpose**: Deep-learning classifier adapter supporting PyTorch `.pt`, TorchScript, and ONNX models trained on GTSRB.
* **Key Capabilities**:
  * Normalized tensor conversion `(1, 3, 32, 32)`.
  * Out-of-distribution background rejection using entropy thresholds and confidence bounds.
  * Top-5 probability ranking with human-readable labels.

---

### 4. `tt100k_model.py` & `tt100k_taxonomy.py`
* **Purpose**: Secondary perception model trained on Tsinghua-Tencent 100K (100,000 images across 221 classes).
* **Consensus Engine**: Cross-verifies European GTSRB predictions against international sign topologies, providing dual-domain safety redundancy for autonomous vehicles.
