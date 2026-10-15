# Fine-Grained Traffic Sign Classification Engine (`modules/C_classification/`)

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
   ▼
MaxPool2D (2×2 pool size, stride 2)
   │
   ▼
Conv2D (64 filters, 3×3 kernel, ReLU activation)
   │
   ▼
MaxPool2D (2×2 pool size, stride 2)
   │
   ▼
Flatten
   │
   ▼
Dense (128 neurons, ReLU activation)
   │
   ▼
Dropout (rate = 0.50)
   │
   ▼
Dense (43 neurons, Softmax activation)
   │
   ▼
Class Probabilities: P(y = c | x), c ∈ {0, ..., 42}
```

---

## 🚀 Multi-Framework Runtime Interoperability

To guarantee cross-platform deployment across diverse hardware environments:

1. **Keras 3 Runtime**: Built on Keras 3 with Torch/TensorFlow backend interoperability (`KERAS_BACKEND=torch`).
2. **PyTorch Tensor Compatibility**: Can execute directly through PyTorch tensors with zero native TensorFlow C-library dependency, supporting Windows, Linux, and macOS.
3. **Loss Function**: Sparse Categorical Crossentropy / Cross-Entropy Loss with Adam optimizer ($\eta = 0.001$).

---

## 📈 Performance Benchmarks

Evaluated rigorously across both validation splits and the complete GTSRB benchmark test set:

| Evaluation Metric | Benchmark Result |
| :--- | :--- |
| **Validation Accuracy** | **99.34%** |
| **Official Test Set Accuracy** | **95.20%** |
| **Single-Image Inference Latency** | **4.2 ms** (CPU) / **1.1 ms** (CUDA) |
| **Stop Sign (Class 14) Confidence** | **100.00%** |

---

## 📁 Directory Structure

```text
modules/C_classification/
├── models/                     # Trained classification model binaries
│   └── traffic_sign_model.keras # 43-Class GTSRB trained model (3.88 MB)
├── results/                    # Training histories and visual plots
│   └── plots/
│       ├── accuracy.png        # Training vs validation accuracy curves
│       └── loss.png            # Training vs validation loss convergence
├── src/                        # Standalone classification executables
│   ├── data_split.py           # Stratified train/val data splitting
│   ├── evaluate.py             # Evaluation script for Test.csv
│   ├── predict.py              # Single-image prediction CLI
│   ├── preprocessing.py        # Normalization and image resizing functions
│   └── train.py                # Full training pipeline script
├── classifier.py               # StandaloneGTSRBClassifier implementing BaseClassifier
├── data_loader.py              # Folder-based dataset loader
├── evaluate.py                 # Multi-backend evaluation interface
├── predict.py                  # High-level predict_sign(path_or_array) interface
├── train_classifier.py         # PyTorch GTSRB classifier training routine
├── test_module_c.py            # Automated pytest quality suite
└── README.md                   # Subsystem technical documentation
```

---

## 💻 CLI Usage & Standalone Scripts

### 1. Single Image Classification (CLI)
```bash
python modules/C_classification/src/predict.py data/samples/class_14_sample_1.png
```

### 2. Model Evaluation on GTSRB Test Set
```bash
python modules/C_classification/src/evaluate.py
```

### 3. Training the Model from Scratch
```bash
python modules/C_classification/src/train.py
```

---

## 🔌 Programmatic Integration

### High-Level Function Interface
```python
from modules.C_classification.predict import predict_sign

# Predict directly from an image file path or numpy BGR array
class_id, confidence = predict_sign("data/samples/class_14_sample_1.png")
print(f"Predicted Class ID: {class_id}, Confidence: {confidence:.2%}")
```

### Object-Oriented Classifier Interface
```python
from modules.C_classification.classifier import StandaloneGTSRBClassifier

classifier = StandaloneGTSRBClassifier()
result = classifier.classify(crop)
print(f"Sign: {result.class_name} ({result.confidence:.1%}) | Category: {result.category.value}")
```

---

## 🧪 Unit Testing

Execute automated validation covering input dimensions, tensor shapes, BGR channel order, and prediction confidence:
```bash
pytest modules/C_classification/test_module_c.py -v
```
