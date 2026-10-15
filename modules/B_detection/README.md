# Real-Time Traffic Sign Detection Interface

This module provides a standalone object detection interface conforming to the `BaseDetector` contract, facilitating localized bounding box extraction and YOLOv8 inference.

---

## 📌 Architecture & Design Contract

The detection interface exposes a standardized, decoupled method signature ensuring compatibility across different detection backends:

```python
from src.detection.base import BaseDetector
from src.schema import DetectionResult, BoundingBox

class StandaloneYOLODetector(BaseDetector):
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
        """
        Executes real-time sign localization on a single frame.

        Args:
            image: BGR numpy uint8 array (H, W, 3).
            conf_threshold: Minimum confidence threshold (0.0 to 1.0).

        Returns:
            List[DetectionResult]: Detected signs with bounding box coordinates,
                                   confidence scores, and source labels.
        """
```

---

## ⚙️ Key Technical Features

1. **Aspect-Ratio & Boundary Clamping**:
   * Enforces coordinate bounds $[0 \le x_1 < x_2 \le W, 0 \le y_1 < y_2 \le H]$ to prevent array slicing exceptions.
   * Filters out extreme aspect ratios to suppress false detections on utility poles or road markings.
2. **Scale Guarding**:
   * Ingested images with $\max(W, H) \le 160\text{px}$ are treated as pre-cropped sign patches, preventing the detector from sub-cropping internal sign symbols.
3. **Weight Resolution**:
   * Automatically resolves canonical weights at `weights/detection/best.pt` or `detection/models/best.pt`.
   * Seamlessly falls back to heuristic shape detection if neural weights are not present.

---

## 💻 Standalone Verification

Execute unit tests verifying detection contracts, boundary handling, and confidence filtering:
```bash
python -m pytest modules/B_detection/test_module_b.py -v
```
