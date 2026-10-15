# System Architecture & Technical Specifications

**Road Traffic Sign Detection & Recognition System**  
**Integration Lead (Member D)**: Sombuddha Ash  
**Dataset**: German Traffic Sign Recognition Benchmark (GTSRB, 43 Classes)  

---

## 1. Domain Decomposition & Team Roles

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
│  - Fine-Tuned YOLOv8n (models/detection/best.pt)                       │
│  - Dynamic CLAHE Pre-conditioning (L channel adaptive equalization)   │
│  - Multi-Scale Letterboxing (640x640 preserving aspect ratio)          │
│  - Non-Maximum Suppression (IoU = 0.45, Conf = 0.25)                   │
│  - Heuristic Color & Geometric Shape Fallback Detector                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Bounding Boxes [x1, y1, x2, y2]
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Member C: Fine-Grained Classification                 │
│  - 43-Class Convolutional Neural Network (traffic_sign_model.keras)    │
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
│  - Pytest Automated Quality Suite (15 Test Cases, 100% Pass)           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Inter-Module Data Contracts

### 2.1 Bounding Box Schema (`src/schema.py`)
```python
@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, width: int, height: int) -> "BoundingBox": ...
    def pad(self, padding: int, width: int, height: int) -> "BoundingBox": ...
    def crop(self, image: np.ndarray, padding: int = 0) -> Optional[np.ndarray]: ...
```

### 2.2 Detection Result Contract (Member B -> Member D)
```python
@dataclass
class DetectionResult:
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    detector_source: str = "YOLOv8"
```

### 2.3 Classification Result Contract (Member C -> Member D)
```python
@dataclass
class ClassificationResult:
    class_id: int
    class_name: str
    confidence: float
    category: SignCategory
    top_k: List[Tuple[int, str, float]]
    speed_limit: Optional[int]
```

### 2.4 End-to-End Perception Contract (Member D -> Consumer / UI / API)
```python
@dataclass
class IntegratedDetection:
    detection: DetectionResult
    classification: Optional[ClassificationResult]
    final_class_id: int
    final_class_name: str
    final_confidence: float
    track_id: Optional[int]

@dataclass
class PipelineFrameResult:
    frame_index: int
    original_shape: Tuple[int, int, int]
    detections: List[IntegratedDetection]
    active_speed_limit: Optional[int]
    inference_time_ms: float
    annotated_frame: Optional[np.ndarray]
```

---

## 3. 43 GTSRB Categories & Mappings

1. **PROHIBITORY Signs (Red Ring / Octagon)**:
   - Speed Limits: 20, 30, 50, 60, 70, 80, 100, 120 km/h (IDs: 0, 1, 2, 3, 4, 5, 7, 8)
   - Passing Restrictions: No passing, No passing vehicles >3.5t (IDs: 9, 10)
   - Vehicle Restrictions: No vehicles, Vehicles >3.5t prohibited, No entry (IDs: 15, 16, 17)
2. **DANGER Signs (Red Triangle)**:
   - General caution, Curves (Left, Right, Double), Bumpy road, Slippery road (IDs: 18-23)
   - Road narrows, Road work, Traffic signals, Pedestrians, Children, Bicycles, Ice/Snow, Wild animals (IDs: 24-31)
3. **MANDATORY Signs (Blue Disc)**:
   - Turn directions: Turn right, Turn left, Ahead only, Straight or right, Straight or left (IDs: 33-37)
   - Directional arrows: Keep right, Keep left, Roundabout mandatory (IDs: 38-40)
4. **OTHER Signs**:
   - Priority & Derestriction: End speed limit 80, Right of way, Priority road, Yield, Stop, End all restrictions, End no passing (IDs: 6, 11, 12, 13, 14, 32, 41, 42)
