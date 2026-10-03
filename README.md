# Road & Traffic Sign Detection and Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-GTSRB%20(43%20Classes)-orange.svg)](https://benchmark.ini.rub.de/)

A modular, real-time Computer Vision and Deep Learning integration pipeline for detecting, recognizing, and tracking German Traffic Signs (GTSRB 43 classes) and custom regional datasets. Built and maintained by **Member D (Integration & Pipeline Lead)**.

---

## 🌟 Key Capabilities & Testing Options

The interactive dashboard (`app.py`) provides 5 dedicated testing modes:

1. **📊 Option 1: Test Against Established Samples**:
   - Evaluate against official GTSRB benchmark test samples with ground truth and class distribution metrics.
2. **🖼️ Option 2: Test Against Still Images**:
   - Drag and drop any image file (`.png`, `.jpg`, `.webp`) from your local disk for instant localized bounding boxes and confidence scores.
3. **📹 Option 3: Continuous Real-Time Video via Camera**:
   - Connects to your live USB webcam or car dashcam for continuous video streaming with persistent tracking and ADAS cockpit HUD.
4. **🖥️ Option 4: Continuous Video via Screenshare (Laptop Video Detection)**:
   - Ultra-fast real-time screen capture using `mss`. Play any video on your laptop (dashcam footage, media player, or browser) and the system watches your screen and detects traffic signs continuously!
5. **🌐 Option 5: Test Against YouTube Video Links**:
   - Paste any YouTube driving video URL (e.g. Autobahn dashcam). Streams directly via `yt-dlp` or downloads a short clip for local high-speed playback.
6. **⏱️ Live Timestamped Event Log**:
   - Real-time event ticker recording every detected sign with the user's local timestamp (e.g. `03/10/2026 16:20:05 -> Stop sign detected`).
   - One-click export to **CSV** or **JSON** for driver trip telemetry.
7. **📁 Custom Dataset Support & On-the-Fly Training**:
   - Upload custom datasets (ZIP folder-per-class) for regional signs (e.g. Indian, US MUTCD, or industrial).
   - One-click PyTorch CNN training with live progress reporting and immediate pipeline deployment.

---

## 🚀 One-Click Run (Self-Hosted)

### Windows Quick Launcher
Double-click `run.bat` or run:
```powershell
.\run.bat
```

### Cross-Platform Python Menu
```powershell
python start.py
```

### Manual Launch
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Streamlit Web Dashboard
streamlit run app.py

# 3. Launch FastAPI REST Service (Optional)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📁 Repository Structure

```text
GFGgemini/
├── run.bat                          # One-click Windows runner
├── start.py                         # Cross-platform CLI interactive menu
├── Dockerfile                       # Containerized self-hosted deployment
├── app.py                           # Streamlit interactive web dashboard
├── api.py                           # FastAPI REST microservice
├── requirements.txt                 # Project dependencies
├── TEAM_INTEGRATION_GUIDE.md        # Integration handbook for Members B & C
├── data/
│   ├── samples/                     # Extracted GTSRB test sample images & videos
│   ├── meta/                        # GTSRB reference icons
│   └── custom/                      # User custom datasets
├── weights/
│   ├── detection/best.pt            # Member B YOLO weights
│   └── classification/classifier.pt # Trained PyTorch CNN weights (99.2% Acc)
├── src/
│   ├── schema.py                    # Data contracts (DetectionResult, ClassificationResult)
│   ├── gtsrb_classes.py             # 43 GTSRB class mapping & category colors
│   ├── pipeline.py                  # Master TrafficSignPipeline orchestrator
│   ├── train_custom.py              # PyTorch CNN model definition & training engine
│   ├── live_feed.py                 # High-speed continuous driving stream runner
│   ├── tracking/
│   │   └── tracker.py               # TemporalSignTracker (IoU matching, smoothing, ADAS state)
│   ├── detection/
│   │   ├── base.py                  # BaseDetector abstract contract
│   │   ├── mock.py                  # MockDetector & ColorContourDetector
│   │   └── yolo.py                  # YOLO detector adapter
│   ├── classification/
│   │   ├── base.py                  # BaseClassifier abstract contract
│   │   ├── mock.py                  # MockClassifier & ColorHeuristicClassifier
│   │   └── model.py                 # PyTorch / ONNX classifier adapter
│   ├── dataset/
│   │   └── custom_dataset.py        # Custom dataset manager & class indexer
│   └── utils/
│       ├── visualizer.py            # High-tech HUD, bounding boxes & ADAS dials
│       ├── video.py                 # Video file & camera stream handler
│       ├── screen_capture.py        # High-FPS screen grabber for video playback
│       ├── youtube.py               # YouTube stream handler via yt-dlp
│       └── event_logger.py          # Local timestamp event logger & CSV/JSON exporter
└── tests/
    ├── test_contracts.py            # Schema & boundary unit tests
    ├── test_adapters.py             # Mock & heuristic adapter tests
    ├── test_pipeline.py             # End-to-end integration tests
    ├── test_tracking.py             # Temporal tracker and ADAS state tests
    ├── test_logger_and_custom.py    # Event logger and custom dataset tests
    └── test_api.py                  # FastAPI endpoint tests
```

---

## 🧪 Testing

Run all 23 automated unit and integration tests:
```powershell
python -m pytest tests/ -v
```
