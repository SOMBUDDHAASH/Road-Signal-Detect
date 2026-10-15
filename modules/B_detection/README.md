# Real-Time Traffic Sign Detection Engine (`modules/B_detection/`)

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
* **Weights**: Located at `modules/B_detection/models/best.pt` (6.25 MB) and `weights/detection/best.pt`.
* **Person & Background Filter**: Strict non-sign class rejection prevents human bodies, faces, or urban clutter from generating false bounding boxes.

### 2. Heuristic Fallback Engine: `ShapeContourDetector`
* **HSV Color Thresholding**: Dual red-hue bands ($[0, 10] \cup [170, 180]$), blue mandatory band ($[100, 135]$), and yellow warning band ($[15, 35]$).
* **Morphological Filtering**: Elliptical kernel opening and closing to suppress environmental noise.
* **Geometric Regularization**: Enforces strict circularity, triangularity, and aspect ratio priors ($0.6 \le W/H \le 1.6$).

---

## 📁 Directory Structure

```text
modules/B_detection/
├── dataset/                    # Validation template splits (images & labels)
│   ├── images/val/.gitkeep
│   └── labels/val/.gitkeep
├── models/                     # Trained detection neural weights
│   └── best.pt                 # Fine-tuned YOLOv8n model (6.25 MB)
├── results/                    # Detection evaluation artifacts & visualizations
│   ├── plots/                  # Precision-recall curves & confidence distributions
│   └── output/                 # Output detections from run_detection.py
├── src/                        # Standalone detection executables
│   ├── detect_yolo.py          # YOLO inference core & camera hooks
│   ├── evaluate_yolo.py        # mAP validator & metric plotter
│   ├── preprocess.py           # Contrast enhancement & letterbox scaling
│   ├── run_detection.py        # CLI interface for images, videos, and live webcam
│   └── train_yolo.py           # End-to-end GTSRB YOLO fine-tuning pipeline
├── detector.py                 # StandaloneYOLODetector adapter implementing BaseDetector
├── requirements.txt            # Detection subsystem dependencies
├── test_module_b.py            # Automated pytest quality suite
└── README.md                   # Subsystem technical documentation
```

---

## 💻 CLI Usage & Standalone Scripts

### 1. Single Image Detection
```bash
python modules/B_detection/src/run_detection.py --source image --input data/samples/class_14_sample_1.png
```

### 2. Video Stream Detection
```bash
python modules/B_detection/src/run_detection.py --source video --input data/samples/simulated_driving.mp4
```

### 3. Model Evaluation on Validation Set
```bash
python modules/B_detection/src/evaluate_yolo.py
```

### 4. Training on Custom or Full GTSRB Data
```bash
python modules/B_detection/src/train_yolo.py --epochs 50 --imgsz 640 --batch 16
```

---

## 🔌 Programmatic Integration

```python
from modules.B_detection.detector import StandaloneYOLODetector

# Automatically resolves weights from modules/B_detection/models/best.pt or weights/
detector = StandaloneYOLODetector()

# Detect traffic signs in an image frame
detections = detector.detect(frame, conf_threshold=0.25)
for det in detections:
    print(f"Detected: {det.class_name} [{det.bbox.to_xyxy()}] conf={det.confidence:.2f}")
```

---

## 🧪 Unit Testing

Execute automated validation covering coordinate clamping, scale guards, and confidence filtering:
```bash
pytest modules/B_detection/test_module_b.py -v
```
