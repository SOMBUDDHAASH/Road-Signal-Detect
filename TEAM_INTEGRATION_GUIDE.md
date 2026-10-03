# Team Integration Guide: Road Traffic Sign Detection & Recognition

**Prepared by**: Sombuddha Ash (Member D — Integration & Pipeline Lead)  
**Primary Dataset**: GTSRB (43 Classes)  
**Branch**: `feature/integration`

---

## 1. Team Responsibilities & Branches

| Member | Role | Branch | Key Deliverable |
| :--- | :--- | :--- | :--- |
| **Member A (Yuvraj Singh)** | Data & Preprocessing | `feature/data-preprocessing` | Cleaned data, train/val/test splits, augmentations |
| **Member B (Shaurya Vaid)** | Detection Lead | `feature/detection` | `detect()` function + YOLO weights (`weights/detection/best.pt`) |
| **Member C (Sneha Chakraborty)** | Classification Lead | `feature/classification` | `classify()` function + CNN weights (`weights/classification/classifier.pt`) |
| **Member D (Sombuddha Ash)** | Integration Lead | `feature/integration` | Master pipeline, Web Dashboard (`app.py`), API (`api.py`), contracts & tests |

---

## 2. Agreed Function Signatures & Contracts

All teammates must adhere to the data contracts defined in [`src/schema.py`](file:///src/schema.py).

### Member B: Detection Contract (`src/detection/base.py`)
```python
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.schema import DetectionResult, BoundingBox

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[DetectionResult]:
        """
        Args:
            image: np.ndarray (H, W, 3) in BGR format
            conf_threshold: float between 0.0 and 1.0
        Returns:
            List[DetectionResult] where each result contains:
                bbox: BoundingBox(x1, y1, x2, y2)
                confidence: float
                detector_label: str (default: "traffic_sign")
        """
        pass
```

**Where to deliver:**
1. Put trained model file at: `weights/detection/best.pt`
2. Implement or verify your detector in `src/detection/yolo.py`.

---

### Member C: Classification Contract (`src/classification/base.py`)
```python
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.schema import ClassificationResult, SignCategory

class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, crop: np.ndarray) -> ClassificationResult:
        """
        Args:
            crop: np.ndarray (H, W, 3) cropped traffic sign image (BGR or RGB)
        Returns:
            ClassificationResult:
                class_id: int (0 to 42 for GTSRB)
                class_name: str (e.g. "Stop", "Speed limit (30km/h)")
                confidence: float (0.0 to 1.0)
                category: SignCategory (PROHIBITORY, DANGER, MANDATORY, OTHER)
                top_k: List[Tuple[int, str, float]] (optional top-3 or top-5 predictions)
        """
        pass
```

**Where to deliver:**
1. Put trained PyTorch weights at: `weights/classification/classifier.pt`
2. Implement or verify your model loader in `src/classification/model.py`.

---

## 3. How to Test Your Code with the Pipeline

Before submitting a PR, test your module directly:

```powershell
# 1. Run unit and integration tests
python -m pytest tests/ -v

# 2. Run the pipeline on a test image with your model
python -m src.pipeline --input data/samples/00000.png --output output.jpg --mode production

# 3. Launch the interactive web app to inspect visual detections
streamlit run app.py
```

If your weights are not trained yet, you can test with the live heuristic fallback:
```powershell
python -m src.pipeline --input data/samples/00000.png --output output.jpg --mode heuristic
```
