# Module A: Data Preprocessing & Augmentation Pipeline

**Module Owner:** Member A (Yuvraj Singh)  
**Git Branch:** `feature/data-preprocessing`  
**Downstream Consumers:** Member C (Classifier training), Member B (YOLO dataset formatting), Member D (Integration pipeline)

---

## 1. Overview & Current Reference Implementation

Member D (Integration Lead) has provided a fully operational, benchmark-compatible baseline in this directory:
- `data_pipeline.py`: Contains `GTSRBDataPipeline`, which resizes, normalizes, converts BGR to RGB, transposes to `(C, H, W)`, performs data augmentations (rotation, brightness/contrast jitter, Gaussian blur), and generates Exploratory Data Analysis (EDA) class imbalance metrics.
- `dataset_downloader.py`: Scans, validates, and extracts GTSRB dataset archives into `data/` (`Train/`, `Test/`, `Meta.csv`, `Train.csv`, `Test.csv`).
- `test_module_a.py`: Pytest suite verifying contract adherence.

---

## 2. API Contract (What Your Code Must Deliver)

When you replace or enhance the baseline with your own preprocessing methods, ensure the following interfaces remain consistent:

### Preprocessing Output
```python
def preprocess_image(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Args:
        image: BGR numpy uint8 image of shape (H, W, 3).
        normalize: If True, values in [0.0, 1.0] (float32).
    Returns:
        numpy ndarray of shape (3, target_h, target_w), default (3, 32, 32).
    """
```

### Dataset Layout in `data/`
Your preprocessing script or downloader should produce or expect:
```
data/
├── Meta.csv
├── Train.csv
├── Test.csv
├── Meta/
│   ├── 0.png ... 42.png
├── Train/
│   ├── 0/ ... 42/
└── Test/
    ├── 00000.png ...
```

---

## 3. Step-by-Step Instructions: How to Swap In Your Own Work

1. **Checkout your branch:**
   ```bash
   git checkout feature/data-preprocessing
   ```

2. **Add your custom augmentations / filters:**
   - You can edit `modules/A_data_preprocessing/data_pipeline.py` or create your own script (e.g. `yuvraj_preprocessing.py`).
   - If you want to introduce **CLAHE** (Contrast Limited Adaptive Histogram Equalization) or **Albumentations**, implement it inside `preprocess_image` or `augment_image`.

3. **Verify contracts with tests:**
   ```bash
   python -m pytest modules/A_data_preprocessing/test_module_a.py -v
   ```

4. **Integration with Member C:**
   Member C trains the PyTorch classifier using `src/train_custom.py` and expects images shaped `(B, 3, 32, 32)` normalized to `[0.0, 1.0]`. If you change the input resolution (e.g., to `(64, 64)`), coordinate with Member C and update `src/schema.py` and `src/classification/model.py`.

5. **Commit and submit:**
   ```bash
   git add modules/A_data_preprocessing/
   git commit -m "feat(data): implement Member A custom preprocessing and augmentation"
   git push origin feature/data-preprocessing
   ```

---

## 4. Production Engineering Upgrades (Required for Real Pipeline)

GTSRB models often fail on real-world inputs because they lack exposure to environmental degradations and scale variations. When deploying the real production dataset, Member A must implement:

1. **Adverse Weather Augmentation**:
   - Integrate `Albumentations` to simulate realistic driving degradations.
   - Implement synthetic rain streaks, night luminance reduction ($\gamma \in [0.35, 0.70]$), and vehicle motion blur (kernel sizes $5 \times 5$ to $13 \times 13$).
2. **Class Imbalance Correction**:
   - GTSRB has extreme frequency variance (Class 2 has 2,250 samples; Class 0 has only 210 samples).
   - Apply SMOTE or random oversampling with geometric jitter to balance minority classes (e.g. 20 km/h, dangerous curve left) to at least 800 training samples.
3. **Multi-Scale Resolution Bucketing**:
   - Store and train crops across variable resolutions ($16 \times 16$ up to $128 \times 128$ pixels) to mirror actual dashcam distances and avoid pixelation distortion.

