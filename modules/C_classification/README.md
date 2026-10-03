# Module C: GTSRB Traffic Sign Classification (43 Classes)

**Module Owner:** Member C (Sneha Chakraborty)  
**Git Branch:** `feature/classification`  
**Downstream Consumers:** Member D (Integration Pipeline, Temporal Tracker, UI HUD, FastAPI)

---

## 1. Overview & Current Reference Implementation

Member D (Integration Lead) has provided a fully operational, high-accuracy baseline in this directory:
- `classifier.py`: `StandaloneGTSRBClassifier` class implementing `BaseClassifier`. Handles input cropping, 32x32 resizing, BGR-to-RGB conversion, normalization, inference, top-k ranking, and out-of-distribution / background rejection (`min_confidence=0.65`).
- `train_classifier.py`: Complete PyTorch CNN training routine that trains on GTSRB 43 classes and exports a deployable TorchScript model.
- `evaluate.py`: Benchmark evaluation utility calculating Top-1 accuracy, Top-5 accuracy, and per-category precision/recall against GTSRB benchmark samples.
- `weights/classifier.pt`: Working pretrained CNN model trained on GTSRB with **99.22% validation accuracy**.
- `test_module_c.py`: Pytest test suite testing contracts, input dimensions, and edge cases.

---

## 2. API Contract (What Member D's Pipeline Expects)

Your classifier must conform to `BaseClassifier` from `src.classification.base`:
```python
from src.classification.base import BaseClassifier
from src.schema import ClassificationResult, SignCategory

class YourClassifier(BaseClassifier):
    def classify(self, crop: np.ndarray) -> ClassificationResult:
        """
        Args:
            crop: BGR numpy uint8 image patch of the detected sign (H, W, 3).
        Returns:
            ClassificationResult containing:
              - class_id: int (0 to 42, or -1 if non-sign/background)
              - class_name: str (human-readable sign name from GTSRB)
              - confidence: float (0.0 to 1.0)
              - category: SignCategory (PROHIBITORY, DANGER, MANDATORY, OTHER)
              - top_k: Optional list of (class_id, class_name, confidence)
        """
```

---

## 3. Step-by-Step Instructions: How to Swap In Your Own Model

### Option A: Direct Weights Drop-in (Easiest)
If you trained a CNN, ResNet, MobileNetV3, or Vision Transformer:
1. Export your model as a TorchScript model (`torch.jit.trace` or `torch.jit.script`) or standard PyTorch state dict.
2. Save or copy your weights file to:
   - `modules/C_classification/weights/classifier.pt`
   - `weights/classification/classifier.pt`
3. Run the evaluation script to verify accuracy:
   ```bash
   python modules/C_classification/evaluate.py --samples 50
   ```
4. Run the module unit tests:
   ```bash
   python -m pytest modules/C_classification/test_module_c.py -v
   ```

### Option B: Custom Neural Network Architecture
If your model requires custom preprocessing (e.g. 48x48 resolution, CLAHE, or specific normalization tensors):
1. Open `modules/C_classification/classifier.py`.
2. Update `preprocess(self, crop)` to match your architecture's requirements.
3. Update `classify(self, crop)` if your logits require custom calibration.

---

## 4. How to Train Your Model Using `train_classifier.py`

You can train or retrain your classifier with:
```bash
python modules/C_classification/train_classifier.py --epochs 10 --batch-size 64 --lr 0.0015 --samples 150
```
Upon completion, the trained model is automatically saved to both `modules/C_classification/weights/classifier.pt` and `weights/classification/classifier.pt`.

---

## 5. Verification & Submission

1. **Run tests:**
   ```bash
   python -m pytest modules/C_classification/test_module_c.py -v
   python -m pytest tests/test_classification.py -v
   ```
2. **Launch Streamlit Dashboard to verify predictions:**
   ```bash
   streamlit run app.py
   ```
3. **Commit your changes:**
   ```bash
   git add modules/C_classification/
   git commit -m "feat(classification): integrate Member C trained classifier"
   git push origin feature/classification
   ```
