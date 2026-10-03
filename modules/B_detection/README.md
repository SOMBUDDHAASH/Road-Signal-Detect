# Module B: Real-Time Traffic Sign Detection

**Module Owner:** Member B (Shaurya Vaid)  
**Git Branch:** `feature/detection`  
**Downstream Consumers:** Member D (Integration pipeline, Tracker, Web/API Dashboards)

---

## 1. Overview & Current Reference Implementation

Member D (Integration Lead) has supplied a fully functional detection implementation in this directory:
- `detector.py`: `StandaloneYOLODetector` class implementing `BaseDetector` from `src.detection.base`. It performs YOLOv8 inference, boundary clamping, aspect ratio filtering, and returns standardized `DetectionResult` objects.
- `train_yolo.py`: Full Ultralytics training script that trains YOLO on traffic sign datasets and automatically copies `best.pt` into the weights directories.
- `weights/best.pt`: Working pretrained YOLOv8 model from the internet ready for testing.
- `test_module_b.py`: Pytest test suite testing contracts, edge cases, and confidence thresholds.

---

## 2. API Contract (What Member D's Pipeline Expects)

Your detector must conform to `BaseDetector`:
```python
from src.detection.base import BaseDetector
from src.schema import DetectionResult, BoundingBox

class YourDetector(BaseDetector):
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
        """
        Args:
            image: BGR numpy uint8 image frame (H, W, 3).
            conf_threshold: Minimum confidence score (0.0 to 1.0).
        Returns:
            List of DetectionResult containing:
              - bbox: BoundingBox(x1, y1, x2, y2)
              - confidence: float (0.0 to 1.0)
              - detector_label: str (e.g. "traffic_sign")
              - detector_class_id: int
        """
```

---

## 3. Step-by-Step Instructions: How to Swap In Your Fine-Tuned Model

### Option A: Direct Weights Drop-in (Easiest & Recommended)
If you trained a standard YOLO model (YOLOv8n, YOLOv8s, YOLOv11):
1. Copy your trained `best.pt` file into:
   - `modules/B_detection/weights/best.pt`
   - `weights/detection/best.pt`
2. Run tests to confirm it loads:
   ```bash
   python -m pytest modules/B_detection/test_module_b.py -v
   ```
3. Launch the full application to see your detector in real-time:
   ```bash
   streamlit run app.py
   ```

### Option B: Custom Model Architecture or Algorithm
If you are using Faster R-CNN, SSD, RT-DETR, or custom post-processing:
1. Edit `modules/B_detection/detector.py`.
2. Ensure your class implements `detect(self, image: np.ndarray, conf_threshold: float)` and outputs `List[DetectionResult]`.
3. Update `src/detection/yolo.py` or import your class in `src/pipeline.py`.

---

## 4. How to Train Your Model Using `train_yolo.py`

You can use the included training utility:
```bash
python modules/B_detection/train_yolo.py --data data/detection_data.yaml --epochs 30 --imgsz 640 --batch 16
```
Upon completion, the script automatically copies your best weights to both `modules/B_detection/weights/` and `weights/detection/`.

---

## 5. Verification & Submission

1. **Run tests:**
   ```bash
   python -m pytest modules/B_detection/test_module_b.py -v
   python -m pytest tests/test_detection.py -v
   ```
2. **Commit your changes:**
   ```bash
   git add modules/B_detection/
   git commit -m "feat(detection): integrate Member B fine-tuned YOLO detector"
   git push origin feature/detection
   ```

---

## 6. Production Engineering Upgrades (Required for Real Pipeline)

High-confidence misclassifications often occur because the detector passes a poorly cropped image (e.g., clipping part of the sign or capturing background clutter). When deploying the production detector, Member B must implement:

1. **Small-Object Anchor Tuning**:
   - Standard YOLO anchor priors are biased toward pedestrians and vehicles.
   - Optimize YOLO anchor box priors specifically for objects below $32 \times 32$ pixels using k-means clustering on GTSDB ground-truth bounding boxes.
2. **Aspect-Ratio Clamping**:
   - Enforce strict square-ish aspect ratio priors ($0.75 \le W/H \le 1.33$) to suppress false detections on vertical structures like utility poles, building edges, and trees.
3. **Scale-Guarded Passthrough Integration**:
   - Ensure the pipeline handles pre-cropped inputs differently from full scene frames.
   - Maintain Member D's scale guard (treating inputs with $\max(W,H) \le 160\text{px}$ as direct signs to prevent YOLO from sub-cropping internal icons).

