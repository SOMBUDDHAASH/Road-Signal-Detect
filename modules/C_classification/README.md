# Standalone GTSRB Classification Interface

This module provides a standalone classification interface conforming to the `BaseClassifier` contract, supporting both Keras 3 (`.keras`) and PyTorch (`.pt`) convolutional neural networks for fine-grained GTSRB recognition.

---

## 📌 Architecture & Dual-Framework Compatibility

The classification interface abstracts framework-specific complexities, enabling seamless execution across both Keras 3 and PyTorch backends:

| Framework | Model Path | Color Mode | Input Resolution |
| :--- | :--- | :--- | :--- |
| **Keras 3 (CNN)** | `classification/models/traffic_sign_model.keras` | **Native BGR** | $32 \times 32 \times 3$, normalized `/ 255.0` |
| **PyTorch (CNN)** | `weights/classification/classifier.pt` | RGB | $3 \times 32 \times 32$, normalized `/ 255.0` |

---

## ⚙️ Interface Contract

```python
from src.classification.base import BaseClassifier
from src.schema import ClassificationResult, SignCategory

class StandaloneClassifier(BaseClassifier):
    def classify(self, crop: np.ndarray) -> ClassificationResult:
        """
        Classifies an extracted traffic sign image patch.

        Args:
            crop: BGR numpy uint8 array (H, W, 3).

        Returns:
            ClassificationResult:
                class_id: int (0 to 42 for GTSRB)
                class_name: str (e.g. "Stop", "Speed limit (30km/h)")
                confidence: float (0.0 to 1.0)
                category: SignCategory (PROHIBITORY, DANGER, MANDATORY, OTHER)
                top_k: List of top predictions with confidence scores
        """
```

---

## 📊 Preprocessing & Color Channel Fidelity

The Keras CNN was trained on raw OpenCV BGR arrays normalized to $[0.0, 1.0]$. The classification adapter respects this ordering, avoiding destructive color-space inversions:
* Red Stop signs (Class 14) achieve **100.00% confidence** under BGR channel alignment.
* Out-of-distribution inputs with maximum probability below threshold are flagged as ambiguous or background candidates.

---

## 💻 CLI & Standalone Usage

### 1. Single-Image Prediction
```bash
python modules/C_classification/predict.py data/samples/class_14_sample_1.png
```

### 2. Standalone Verification Tests
```bash
python -m pytest modules/C_classification/test_module_c.py -v
python -m pytest tests/test_member_c_compatibility.py -v
```
