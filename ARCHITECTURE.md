# System Architecture & Technical Specifications

**Autonomous Road Traffic Sign Detection & Recognition System**  
**Lead Architect**: Sombuddha Ash  
**Benchmark Datasets**: German Traffic Sign Recognition Benchmark (GTSRB, 43 Classes) & Tsinghua-Tencent 100K (TT100K, 221 Classes)  

---

## 1. Subsystem Architecture & Perception Flow

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Master Input Modalities                         │
│       Still Images • Video Recordings • Live Camera • REST Stream      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              Subsystem 1: Data Engineering & Augmentation              │
│  - Automated Dataset Ingestion & Validation Protocols                  │
│  - Corrupt Image Filtering & Coordinate Boundary Clamping              │
│  - Track-Aware Stratified Train/Val/Test Splitting (Zero Leakage)      │
│  - Orientation-Preserving Augmentations (No Illegal Horizontal Flips)  │
│  - Class Distribution Balancing & Automated EDA Profiling              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             Subsystem 2: Real-Time Sign Detection Engine               │
│  - Fine-Tuned YOLOv8n Primary Detector (modules/B_detection/models/)   │
│  - Dynamic CLAHE Pre-conditioning (L channel adaptive equalization)   │
│  - Aspect-Preserving Multi-Scale Letterboxing (640x640)                │
│  - Non-Maximum Suppression (IoU = 0.45, Conf = 0.25)                   │
│  - Heuristic Color & Geometric Shape Fallback Detector                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Bounding Boxes [x1, y1, x2, y2]
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│          Subsystem 3: Fine-Grained Deep Neural Classification          │
│  - 43-Class Convolutional Neural Network (modules/C_classification/)   │
│  - Multi-Backend Interoperability (PyTorch & Keras 3 Runtimes)         │
│  - Standardized BGR Input Normalization [0, 1] for 32x32x3 Patches     │
│  - Dual-Domain TT100K 221-Class Cross-Domain Consensus Verification    │
│  - 99.34% Validation Accuracy / 95.20% Official GTSRB Test Accuracy    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Classification Predictions & Probs
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│      Subsystem 4: Perception Integration, Tracking & Orchestration     │
│  - End-to-End Multimodal Perception Pipeline Orchestrator              │
│  - Temporal Multi-Object Tracker (SignTrack continuity & hysteresis)   │
│  - Active Speed Limit Memory Zone Handler                              │
│  - High-Resolution ADAS Cockpit HUD Visualizer (Dynamic Pill Badges)   │
│  - Real-time Trip Telemetry Logger (JSON / CSV Exporter)               │
│  - Interactive Streamlit Dashboard (app.py) & FastAPI Server (api.py)  │
│  - Comprehensive Test Suite (68 Automated Pytest Cases, 100% Pass)     │
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

### 2.2 Detection Result Contract (Detection Engine -> Orchestrator)
```python
@dataclass
class DetectionResult:
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    detector_source: str = "YOLOv8"
```

### 2.3 Classification Result Contract (Classifier -> Orchestrator)
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

### 2.4 End-to-End Perception Contract (Orchestration Pipeline -> Application / API)
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
