# Road & Traffic Sign Detection and Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-GTSRB%20(43%20Classes)-orange.svg)](https://benchmark.ini.rub.de/)

A modular, real-time Computer Vision and Deep Learning integration pipeline for detecting and classifying German Traffic Signs (GTSRB 43 classes). Built and maintained by **Member D (Integration & Pipeline Lead)**.

---

## 🌟 Architecture Overview

The system follows a high-performance **two-stage detection-and-classification** architecture:
1. **Localization Stage (Member B)**: Detects traffic sign regions in high-resolution frames using YOLOv8/v11 or HSV color-shape contour heuristics.
2. **Cropping & Normalization (Member D)**: Safely extracts candidate bounding boxes with coordinate clamping and dynamic margin padding.
3. **Recognition Stage (Member C)**: Classifies cropped sign regions into one of 43 GTSRB benchmark classes using a deep Convolutional Neural Network (CNN / MobileNetV3).
4. **Fusion & Telemetry (Member D)**: Overlays category-coded HUD bounding boxes, confidence badges, latency profiling, and real-time FPS estimation.

---

## 🚀 Quickstart

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Prepare Sample Images
```powershell
python scripts/prepare_sample_data.py
```

### 3. Run Pipeline via CLI
```powershell
python -m src.pipeline --input data/samples/00000.png --output output.jpg --mode heuristic
```

### 4. Launch Interactive Web Dashboard
```powershell
streamlit run app.py
```

### 5. Start REST API Server
```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📁 Repository Structure

```text
GFGgemini/
├── app.py                      # Streamlit interactive web dashboard
├── api.py                      # FastAPI REST microservice
├── requirements.txt            # Project dependencies
├── TEAM_INTEGRATION_GUIDE.md   # Integration handbook for Members B & C
├── data/
│   ├── samples/                # Extracted GTSRB test sample images
│   └── meta/                   # GTSRB reference icons
├── weights/
│   ├── detection/              # Member B YOLO weights (best.pt)
│   └── classification/         # Member C CNN weights (classifier.pt)
├── src/
│   ├── schema.py               # Data contracts (DetectionResult, ClassificationResult)
│   ├── gtsrb_classes.py        # 43 GTSRB class mapping & category colors
│   ├── pipeline.py             # Master TrafficSignPipeline orchestrator
│   ├── detection/
│   │   ├── base.py             # BaseDetector abstract contract
│   │   ├── mock.py             # MockDetector & ColorContourDetector
│   │   └── yolo.py             # YOLO detector adapter
│   ├── classification/
│   │   ├── base.py             # BaseClassifier abstract contract
│   │   ├── mock.py             # MockClassifier & ColorHeuristicClassifier
│   │   └── model.py            # PyTorch / ONNX classifier adapter
│   └── utils/
│       ├── visualizer.py       # High-tech HUD drawing & telemetry overlay
│       └── video.py            # Video file & camera stream handler
└── tests/
    ├── test_contracts.py       # Schema & boundary unit tests
    ├── test_adapters.py        # Mock & heuristic adapter tests
    └── test_pipeline.py        # End-to-end integration tests
```

---

## 🧪 Testing

Run all automated unit and integration tests:
```powershell
python -m pytest tests/ -v
```
