# Real-Time Traffic Sign Detection Engine

A high-speed, real-time object detection engine utilizing fine-tuned **YOLOv8** with geometric and HSV color-space fallback mechanisms for robust traffic sign localization across diverse driving environments.

---

## 📌 Subsystem Overview

In autonomous driving perception, the detection subsystem provides primary spatial localization:
1. **Scene Ingestion**: Ingests high-resolution video streams, dashcam footage, or still camera captures.
2. **Contrast & Dynamic Enhancement**: Applies CLAHE pre-conditioning on the luminance channel for challenging lighting (night, glare, rain, fog).
3. **Bounding Box Proposal**: Locates all traffic signs within the visual field with normalized pixel coordinates.
4. **Boundary Guarding**: Enforces strict boundary clamping, aspect-ratio priors, and scale guards to isolate clean regions of interest (ROI) for downstream classification.

---

## 📊 Dataset & Training Specifications

Trained and evaluated on the **German Traffic Sign Detection Benchmark (GTSDB)**:

| Property | Value / Specification |
| :--- | :--- |
| **Dataset Benchmark** | German Traffic Sign Detection Benchmark (GTSDB) |
| **Number of Images** | 900 high-resolution road scene images |
| **Target Classes** | 43 standard European traffic sign categories |
| **Annotation Standard** | Standardized bounding box tuples `[x1, y1, x2, y2, class_id]` |
| **Input Resolution** | $640 \times 640 \times 3$ with aspect-ratio letterboxing |
| **Inference Thresholds** | Detection Confidence: `0.25` • NMS IoU: `0.45` |

---

## 🧠 Dual-Engine Detection Architecture

To guarantee high availability and fault tolerance, the detection module incorporates two complementary engines:

```text
Incoming Road Scene Image (H × W × 3)
                 │
                 ▼
       ┌───────────────────┐
       │ Primary: YOLOv8n  │ ──(Weights Available)──► Bounding Boxes & Confidence
       └─────────┬─────────┘
                 │ (Fallback if missing or degraded)
                 ▼
       ┌────────────────────────┐
       │ Heuristic Color-Shape  │ ──(HSV Thresholds + Contour Circularity)──► Candidate ROIs
       └────────────────────────┘
```

### 1. Primary Engine: Fine-Tuned YOLOv8n
* **Backbone**: Single-stage nano convolutional neural network pre-trained on COCO and fine-tuned on GTSDB.
* **Weights**: Located at `detection/models/best.pt` (6.25 MB).
* **Person & Background Filter**: Strict non-sign class rejection prevents human bodies, faces, or urban clutter from generating false bounding boxes.

### 2. Heuristic Fallback Engine: `ShapeContourDetector`
* **HSV Color Thresholding**: Dual red-hue bands ($[0, 10] \cup [170, 180]$), blue mandatory band ($[100, 135]$), and yellow warning band ($[15, 35]$).
* **Morphological Filtering**: Elliptical kernel opening and closing to suppress environmental noise.
* **Geometric Regularization**: Enforces strict circularity, triangularity, and aspect ratio priors ($0.6 \le W/H \le 1.6$).

---

## 📈 Performance & Latency Metrics

| Metric | Score | Conditions |
| :--- | :--- | :--- |
| **mAP@50** | **88.6%** | GTSDB test evaluation |
| **Inference Latency (GPU)** | **~12 ms** | NVIDIA RTX / TensorRT enabled |
| **Inference Latency (CPU)** | **~38 ms** | Intel / AMD x86_64 CPU |
| **Frame Rate** | **30+ FPS** | Sustained real-time video processing |

---

## 💻 CLI & Usage Instructions

### 1. Single Image Detection
Run detection on a test image and view localized bounding boxes:
```bash
python detection/src/detect_yolo.py --source data/samples/class_14_sample_1.png
```

### 2. Video Stream Detection
Process a driving video file with real-time bounding box annotations:
```bash
python detection/src/run_detection.py --video data/samples/simulated_driving.mp4
```

### 3. Model Training
Train or fine-tune YOLOv8 on custom traffic sign datasets:
```bash
python detection/src/train_yolo.py --epochs 50 --imgsz 640 --batch 16
```

### 4. Benchmark Evaluation
Evaluate detection accuracy metrics across the test split:
```bash
python detection/src/evaluate_yolo.py
```

---

## 🔌 Python API Integration

```python
import cv2
from src.detection.yolo_detector import YOLODetector

# Initialize detector
detector = YOLODetector("detection/models/best.pt")

# Ingest camera frame
frame = cv2.imread("data/samples/class_14_sample_1.png")

# Run detection
detections = detector.detect(frame, conf_threshold=0.25, iou_threshold=0.45)

for det in detections:
    box = det.bbox
    print(f"Sign localized: [{box.x1}, {box.y1}, {box.x2}, {box.y2}] | Conf: {det.confidence:.2f}")
```
