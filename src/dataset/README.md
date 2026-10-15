# Dataset Management & Benchmark Loaders (`src/dataset/`)

This directory houses the dataset verification, ground-truth benchmark mapping, and custom regional dataset parsing subsystems.

---

## 📄 File Index & Specifications

### 1. `benchmark_loader.py`
* **Purpose**: Ingests and parses official GTSRB benchmark annotations (`Meta.csv`, `Test.csv`, `Train.csv`) for ground-truth validation.
* **Key Components**:
  * `GroundTruthSign`: Dataclass containing image dimensions, ground-truth ROI bounding boxes `(roi_x1, roi_y1, roi_x2, roi_y2)`, class ID, semantic name, sign category, shape, and color codes.
  * `BenchmarkDataLoader`: Memory-mapped lookup table linking image filenames to verified ground truth for computing precision, recall, and confusion matrices.

---

### 2. `custom_dataset.py`
* **Purpose**: Provides extensible support for custom regional datasets (Indian, US MUTCD, industrial safety signs) in standard ZIP format.
* **Key Capabilities**:
  * Dynamic archive extraction into `data/custom/<dataset_name>/`.
  * Automatic directory scanning, label encoding, and metadata JSON serialization.
  * Direct tensor loading for on-the-fly model adaptation.
