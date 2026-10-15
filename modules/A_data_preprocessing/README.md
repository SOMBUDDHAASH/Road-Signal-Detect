# Data Preprocessing & Augmentation Subsystem

This module implements automated dataset acquisition, corrupt image filtering, ROI crop normalization, stratified dataset splitting, and semantic-preserving data augmentation for traffic sign recognition.

---

## 📌 Architecture & Responsibilities

The data preprocessing engine prepares standardized, noise-free, and balanced image datasets for deep learning training and validation:

1. **Dataset Acquisition & Ingestion**: Direct loading or automated KaggleHub fetching (`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`).
2. **Corrupt Image & Anomaly Removal**: Verifies header integrity, removes 0-byte or corrupted image files, and validates bounding box coordinate boundaries.
3. **Track-Aware Stratified Splitting**: Splits datasets into Training (70%), Validation (15%), and Test (15%) partitions while ensuring images from the same physical road sign track are never split across train and test (preventing statistical data leakage).
4. **Orientation-Preserving Augmentation**: Balances minority classes using geometric transformations while strictly avoiding horizontal flips.
5. **Automated EDA Reporting**: Generates class frequency distributions and image galleries for exploratory data analysis.

---

## ⚠️ Semantic Augmentation Rule: Zero Horizontal Flips

In standard computer vision (e.g. ImageNet), horizontal reflection is common. In traffic sign recognition, horizontal flipping **inverts semantic meaning**:
* **Directional Reversal**: "Turn Left" (Class 34) becomes "Turn Right" (Class 33); "Keep Right" (Class 38) becomes "Keep Left" (Class 39).
* **Text & Speed Limit Corruption**: Numerals ($20, 30, 50, \dots$) become mirrored, non-existent characters.

**Permitted Augmentations**:
* Slight rotation: $\theta \in [-10^\circ, +10^\circ]$
* Scale variation: $0.90 \times \dots 1.10 \times$
* Brightness and contrast shifts: $\alpha \in [0.8, 1.2], \beta \in [-20, 20]$
* Environmental Gaussian blur: $\sigma \in [0.5, 1.2]$

---

## 📁 Dataset Layout Specification

Standardized directory structure expected by the pipeline:
```text
data/
├── Meta.csv             # Official class index, shapes, and category mappings
├── Train.csv            # Annotations with width, height, and ROI coordinates
├── Test.csv             # Official test set ground-truth annotations
├── raw/                 # Ingested original image files
├── processed/           # Processed and balanced arrays (.npz format)
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── train_balanced.npz
└── samples/             # Canonical reference image samples (Classes 0-42)
```

---

## 💻 CLI & Execution

### 1. Run Complete Data Pipeline
```bash
python run_pipeline.py --target 500
```

### 2. Standalone Cleaning & ROI Extraction
```bash
python -m src.preprocess
```

### 3. Stratified Splitting
```bash
python -m src.split_data
```

### 4. Data Augmentation & Balancing
```bash
python -m src.augment --target 500
```

### 5. Generate EDA Reports & Charts
```bash
python -m src.eda
```
Outputs distribution charts and class galleries to `results/plots/`.
