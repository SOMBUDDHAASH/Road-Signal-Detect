# Fine-Grained Traffic Sign Classification Engine

A high-performance deep convolutional neural network (CNN) for fine-grained classification across all **43 classes** of the German Traffic Sign Recognition Benchmark (GTSRB).

---

## 📌 Subsystem Overview

In the autonomous perception stack, the classification subsystem serves as the fine-grained recognition engine following sign localization:

1. **Detection**: Real-time object detector localizes candidate bounding boxes `[x1, y1, x2, y2]`.
2. **Crop & Normalization**: The candidate region is cropped, boundary-clamped, and preprocessed.
3. **Classification** (*This Module*): Evaluates the patch, generating class posterior probabilities, category grouping, and top-k confidence rankings.
4. **Integration**: Feeds recognized sign states to temporal tracking, speed-limit memory zones, and ADAS cockpit HUD rendering.

---

## 📊 Dataset & Preprocessing Standards

The model is trained on the official **German Traffic Sign Recognition Benchmark (GTSRB)**:

| Metric | Specification |
| :--- | :--- |
| **Number of Classes** | 43 official traffic sign classes |
| **Training Dataset** | 39,209 annotated sign images |
| **Official Test Set** | 12,630 evaluation images |
| **Input Dimensions** | $32 \times 32 \times 3$ (Height, Width, Channels) |
| **Color Space** | **Native BGR** (OpenCV channel order) |
| **Normalization** | Pixel intensities scaled to $[0.0, 1.0]$ (`float32`) |

> **Channel Ordering**: The model is trained directly on raw OpenCV BGR arrays. Preserving native BGR ensures optimal feature activation without chromatic channel swapping.

---

## 🧠 Neural Network Architecture

The classifier employs a deep feed-forward convolutional architecture optimized for fine-grained symbol discrimination:

```text
Input Patch (32 × 32 × 3, BGR float32)
   │
   ▼
Conv2D (32 filters, 3×3 kernel, ReLU activation)
   │
MaxPool2D (2×2 pool size, stride 2)
   │
Conv2D (64 filters, 3×3 kernel, ReLU activation)
   │
MaxPool2D (2×2 pool size, stride 2)
   │
Flatten
   │
Dense (128 neurons, ReLU activation)
   │
Dropout (rate = 0.50)
   │
Dense (43 neurons, Softmax activation)
   │
   ▼
Class Probabilities: P(y = c | x), c ∈ {0, ..., 42}
```

---

## ⚙️ Framework Interoperability (Keras 3 & PyTorch)

The classification subsystem is engineered with **Keras 3**, providing seamless multi-backend execution:
* **PyTorch Backend** (`KERAS_BACKEND=torch`): Enables direct tensor execution on Python 3.10 through 3.14 on Windows and Linux with zero TensorFlow C-binding requirements.
* **Model Serialization**: Stored in modern `.keras` zip format (`classification/models/traffic_sign_model.keras`, 3.88 MB).

---

## 📈 Benchmark Performance

Evaluated against the official GTSRB benchmark:

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Validation Accuracy** | **99.34%** | On stratified held-out validation set |
| **Official GTSRB Test Set** | **95.20%** | Evaluated on all 12,630 test images |
| **Canonical Test Accuracy** | **100.00%** | Evaluated on standardized canonical sign samples |
| **Inference Latency** | **< 3 ms** | GPU / < 12 ms CPU per sign patch |

---

## 💻 CLI & Usage Instructions

### 1. Model Training
To train the CNN from scratch using GTSRB training data:
```bash
python classification/src/train.py
```
Outputs the trained model to `classification/models/traffic_sign_model.keras`.

### 2. Model Evaluation
To evaluate test accuracy and confusion matrix across the test dataset:
```bash
python classification/src/evaluate.py
```

### 3. Single-Image Inference
To run fine-grained inference on any cropped traffic sign image:
```bash
python classification/src/predict.py --input data/samples/class_14_sample_1.png
```

---

## 🔌 Python API Integration

```python
import cv2
from src.classification.keras_classifier import KerasSignClassifier

# Initialize classifier
classifier = KerasSignClassifier()

# Load image (BGR format)
image = cv2.imread("data/samples/class_14_sample_1.png")

# Run inference
result = classifier.classify(image)

print(f"Predicted: {result.class_name} (Class {result.class_id})")
print(f"Confidence: {result.confidence * 100:.2f}%")
print(f"Category: {result.category}")
```
