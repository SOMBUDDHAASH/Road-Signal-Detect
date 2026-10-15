# 🎓 Presentation Guide & Operational Runbook

**Autonomous Road Traffic Sign Detection & Recognition System**  
**Integration & Pipeline Lead (Member D)**: Sombuddha Ash  
**Repository**: `https://github.com/SOMBUDDHAASH/Road-Signal-Detect.git`  
**Datasets**: German Traffic Sign Recognition Benchmark (GTSRB, 43 Classes) + Tsinghua-Tencent 100K (TT100K, 221 Classes)  

---

## 🌟 1. Executive Summary & Project Pitch

> *"Our team built an enterprise-grade, end-to-end autonomous driving perception platform. We localized and classified all 43 GTSRB traffic sign classes with high accuracy, zero data leakage, and sub-50ms latency. The system features a unified perception pipeline, temporal multi-frame tracking, speed limit memory retention, an ADAS Cockpit HUD, environmental pre-conditioning (night/fog dehazing), a cross-domain secondary consensus engine (TT100K), and a full-stack interface (Streamlit Dashboard, FastAPI REST API, and CLI)."*

---

## 👥 2. Team Member Role Decomposition

When presenting, clearly distinguish each team member's specific contribution and how Member D unified them into a cohesive platform:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Master Input Modalities                         │
│       Still Images • Video Recordings • Live Camera • REST Stream      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              Member A: Data Preprocessing & Augmentation               │
│  - Dataset Ingestion (Kagglehub / Direct archive linking)             │
│  - Corrupt Image Removal & Boundary Clamping                           │
│  - Track-Aware Stratified Train/Val/Test Splitting (No Leakage)        │
│  - Orientation-Preserving Augmentations (NO horizontal flips)          │
│  - Class Balancing (train_balanced.npz) & EDA Generation               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Member B: Sign Detection Engine                     │
│  - Fine-Tuned YOLOv8n (modules/B_detection/models/best.pt)             │
│  - Dynamic CLAHE Pre-conditioning (L channel adaptive equalization)   │
│  - Multi-Scale Letterboxing (640x640 preserving aspect ratio)          │
│  - Non-Maximum Suppression (IoU = 0.45, Conf = 0.25)                   │
│  - Heuristic Color & Geometric Shape Fallback Detector                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Bounding Boxes [x1, y1, x2, y2]
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Member C: Fine-Grained Classification                 │
│  - 43-Class CNN (modules/C_classification/models/traffic_sign_model)   │
│  - Keras 3 with Torch/TensorFlow Backend Interoperability              │
│  - Input: 32x32x3 BGR float32 in [0, 1]                                │
│  - Top-k Confidence Ranking & Category Categorization                  │
│  - 99.34% Internal Test Accuracy / 95.20% Official GTSRB Accuracy     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Classification Predictions & Probs
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Member D: System Integration & Orchestration          │
│  - End-to-End TrafficSignPipeline Orchestration                        │
│  - Temporal Multi-Object Tracker (SignTrack continuity)                │
│  - Active Speed Limit Memory Zone Handler                              │
│  - ADAS Cockpit HUD Visualizer (Category Pill Banners & Badges)        │
│  - Real-time Trip Telemetry Logger (JSON / CSV Exporter)               │
│  - Streamlit Interactive Web Application (app.py)                      │
│  - Production FastAPI REST Server (api.py)                             │
│  - Unified Command-Line Interface (cli.py)                             │
│  - Pytest Automated Quality Suite (68 Tests, 100% Pass)                │
└────────────────────────────────────────────────────────────────────────┘
```

| Member | Focus Area | Key Deliverables & Achievements |
| :--- | :--- | :--- |
| **Member A** (Yuvraj) | **Data Preprocessing & Augmentation** | • `modules/A_data_preprocessing/`: Automated download, corrupt image removal, ROI cropping.<br>• Track-aware stratified train/val/test splitting.<br>• Semantic-preserving data augmentation (brightness, rotation, zoom, blur; strictly NO horizontal flips).<br>• Class balancing & automated EDA charts generation. |
| **Member B** (Shaurya) | **Real-Time Detection (YOLOv8)** | • Fine-tuned YOLOv8n model (`modules/B_detection/models/best.pt`).<br>• Standalone YOLO detector & CLI (`modules/B_detection/src/run_detection.py`).<br>• HSV color-mask and contour geometry fallback detector (`ShapeContourDetector`).<br>• Letterboxing & multi-scale bounding box extraction. |
| **Member C** (Sneha) | **Fine-Grained Classification (CNN)** | • Trained 43-class CNN model (`modules/C_classification/models/traffic_sign_model.keras`).<br>• Standalone predictor & evaluation (`modules/C_classification/src/predict.py`).<br>• Top-k prediction ranking & category grouping.<br>• 99.34% internal test accuracy, verified at 100% confidence on canonical benchmarks. |
| **Member D** (Sombuddha) | **Integration, Architecture & Lead** | • Master perception pipeline (`src/integration/pipeline.py` & `src/pipeline.py`).<br>• Temporal tracking with IoU smoothing and speed-limit memory zones.<br>• ADAS Cockpit HUD overlay & trip telemetry logging (CSV/JSON).<br>• Secondary TT100K 221-class cross-domain consensus engine.<br>• Environmental stress pre-conditioning (night/fog dehazing).<br>• Streamlit Web Dashboard (`app.py`), FastAPI REST Service (`api.py`), Unified CLI (`cli.py`), 1-Click Launchers (`start.py`, `run.bat`), and 68 automated unit/integration tests (100% pass rate). |

---

## 🚀 3. Live Demonstration Playbook (Step-by-Step)

During your presentation, follow this 4-act demonstration flow:

### Act 1: Automated Quality & Testing Verification (The Foundation)
Show that the codebase is robust, thoroughly tested, and CI/CD-ready.
```powershell
pytest tests/ -v
```
* **Talking Point**: *"We have 68 automated tests covering data contracts, coordinate clamping, data loaders, detection letterboxing, Keras/PyTorch model parity, environmental stress dehazing, temporal tracking, REST API endpoints, and TT100K consensus logic. 100% pass rate in under 15 seconds."*

---

### Act 2: Interactive Web Dashboard (The Visual Centerpiece)
Launch the Streamlit web application:
```powershell
streamlit run app.py
```
*(Or use 1-click launcher `.\run.bat` or `python start.py` and choose Option 1).*

**Demonstrate the 5 Testing Modes**:
1. **Option 1: GTSRB Benchmark Explorer**:
   - Filter through the 43 official classes.
   - Show canonical samples matching with 100.0% confidence.
   - Show the confusion matrix and per-class precision/recall charts.
2. **Option 2: Test Still Images**:
   - Upload `data/samples/class_14_sample_1.png` (Stop sign) or any other sample from `data/samples/`.
   - Point out the bounding box, badge, category pill ("PROHIBITORY"), and 100% confidence score.
3. **Option 3: Real-Time Camera / Webcam**:
   - Demonstrate camera feed.
   - Point out the anti-false-detection protection: human skin and bodies are rejected to prevent false positives in a room.
4. **Option 4: Laptop Screenshare (Live Video Detection)**:
   - Click "Screenshare Mode". Play a dashcam driving video on your screen.
   - Show the HUD detecting signs at 30+ FPS directly from the screen capture using `mss`.
5. **Option 5: YouTube Video Stream**:
   - Enter a driving video link or use a local simulation.

**Highlight Special Controls in the Sidebar**:
* **TT100K Secondary Consensus**: Toggle it ON to show dual-model cross-validation with Chinese/European sign taxonomy cross-mapping.
* **Environmental Pre-Conditioner**: Toggle Night Expansion or Fog Dehazing.
* **Acoustic / Spoken Alerts**: Demonstrate audio chime / speech warnings.
* **Export Trip Telemetry**: Click "Download Trip Log (CSV/JSON)" to show live driver event logging.

---

### Act 3: Production REST API (Microservices & Integration)
Show how other autonomous vehicle software can interface with the system via standard REST endpoints:
```powershell
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
* Open browser at **`http://127.0.0.1:8000/docs`** (Interactive Swagger UI).
* Execute **`GET /health`**: Returns engine status and active model backends.
* Execute **`POST /predict`** with an image file: Returns structured JSON with bounding boxes, class names, speed limits, and latency.
* Execute **`POST /predict/annotated`**: Returns the annotated image buffer directly for display.

---

### Act 4: Unified Command-Line Runner (Headless & Automation)
Show how the system runs in automated server scripts or headless edge devices:
```powershell
# Single Image Inference
python cli.py --source image --input data/samples/class_14_sample_1.png

# Video Processing with Multi-frame Tracking
python cli.py --source video --input data/samples/simulated_driving.mp4 --output results/output/video_out.mp4
```

---

## 🎯 4. Key Technical Questions & Answers (FAQ for Evaluators)

### Q1: Why did you forbid horizontal flips during data augmentation?
> **Answer**: Traffic signs are directional and semantic. Flipping a "Turn Left" sign (Class 34) creates a "Turn Right" sign (Class 33). Flipping "Keep Right" (Class 38) creates "Keep Left" (Class 39). Flipping speed limit numbers creates unreadable, corrupt symbols. We strictly limited augmentation to orientation-preserving transforms: rotation ($\pm 10^\circ$), zoom ($0.9 - 1.1\times$), brightness shifts, and Gaussian blur.

### Q2: How did you solve the Python 3.14 compatibility issue with TensorFlow?
> **Answer**: TensorFlow does not currently publish Windows wheels for Python 3.14. We engineered Member C's classifier using **Keras 3 with the PyTorch backend** (`os.environ["KERAS_BACKEND"] = "torch"`). This loads `.keras` models directly through PyTorch tensors with zero native TensorFlow C-library dependency, maintaining cross-platform and forward Python compatibility.

### Q3: Why is the color space critical for Member C's model?
> **Answer**: OpenCV loads images in BGR format by default. Member C trained the CNN on normalized BGR arrays ($[0, 1]$). If the image is swapped to RGB, red and blue channels invert—causing a red Stop sign to be misidentified as a blue mandatory sign. By preserving the BGR normalization pipeline, the model achieves **100.00% confidence** on canonical Stop signs.

### Q4: How does Member D's tracking prevent flickering across video frames?
> **Answer**: In single-frame detection, signs can briefly drop below the detection threshold due to motion blur or occlusion. Member D's `SignTracker` implements multi-frame IoU association ($IoU \ge 0.45$) and retains state over a rolling window. Additionally, active speed limits persist in vehicle memory until explicitly overwritten by a new speed limit or a derestriction sign (Class 32).

### Q5: What is the purpose of the TT100K Secondary Consensus Engine?
> **Answer**: In safety-critical Level 4/5 autonomous driving, single-model perception poses a single point of failure. We implemented a secondary 221-class classifier trained on the Tsinghua-Tencent 100K dataset. The consensus engine evaluates cross-domain confidence:
> - **Consensus Verified**: Both models agree on class or category $\to$ High execution confidence.
> - **Domain Discord**: Primary and secondary models disagree $\to$ Trigger caution advisory and telemetry warning.
> When toggled OFF, it adds 0.0 ms latency overhead.

---

## 📁 5. Repository File Navigation Cheatsheet

```text
GFGgemini/
├── start.py                # 1-Click Interactive Menu Launcher (Cross-platform)
├── run.bat                 # 1-Click Windows Batch Launcher
├── cli.py                  # Unified Command-Line Interface
├── run_pipeline.py         # Complete Data & Perception Pipeline Runner
├── app.py                  # Streamlit Interactive Web Application
├── api.py                  # FastAPI Production REST Microservice
├── ARCHITECTURE.md         # Full Technical Architecture & Specifications
├── PRESENTATION_GUIDE.md   # Presentation Script & Evaluator Q&A
├── README.md               # Master Project Documentation
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Full project dependencies
│
├── modules/
│   ├── A_data_preprocessing/  # Member A data pipeline, downloader & tests
│   ├── B_detection/           # Member B fine-tuned YOLOv8, models, dataset, src & CLI
│   └── C_classification/      # Member C 43-class CNN, models, results, src & CLI
│
├── src/
│   ├── config.py           # Master System Configuration (Members A, B, C, D)
│   ├── schema.py           # Unified Data Contracts & Dataclasses
│   ├── pipeline.py         # Enterprise perception pipeline
│   ├── data_pipeline/      # Member A complete data pipeline
│   ├── integration/        # Member D pipeline, tracker, HUD overlay, telemetry
│   ├── classification/     # Keras 3 / PyTorch classifiers & TT100K model
│   ├── detection/          # YOLOv8 detector & shape contour fallback
│   ├── tracking/           # Temporal tracking & ADAS speed state
│   └── utils/              # Environmental CLAHE, audio alerts, screenshare, video
│
├── tests/                  # 16 test suites (68 tests, 100% passing)
├── weights/                # Canonical models for detection & classification
└── data/samples/           # Canonical test samples (Classes 0-42) & demo video
```
