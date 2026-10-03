# Dataset Management & Benchmark Loaders (`src/dataset/`)

This directory houses the dataset parsing, ground-truth verification, and custom dataset management subsystems developed by **Member D (Integration Lead)** and used by **Member A (Data Lead)**.

---

## 📄 File Index & Detailed Descriptions

### 1. `benchmark_loader.py`
* **Purpose**: Parses official GTSRB benchmark ground truth files (`Meta.csv`, `Test.csv`, `Train.csv`) and matches test samples against true labels.
* **Key Components**:
  * `GroundTruthSign`: Dataclass containing `filename`, `width`, `height`, ground-truth ROI coordinates `(roi_x1, roi_y1, roi_x2, roi_y2)`, `class_id`, `class_name`, `category`, `shape_id`, and `color_id`.
  * `BenchmarkDataLoader`: Loads all annotations into memory maps:
    * `get_ground_truth(filename)`: Looks up ground truth for any sample file (e.g. `00000.png` or `class_14_sample_1.png`).
* **Interconnections**: Used by Option 1 in `app.py` and `modules/C_classification/evaluate.py` to calculate exact benchmark precision and recall.

---

### 2. `custom_dataset.py`
* **Purpose**: Allows users and enterprise teams to upload custom regional datasets (Indian, US MUTCD, industrial signs) in ZIP format and train on them dynamically.
* **Key Components**:
  * `CustomDatasetManager`:
    * Unpacks ZIP files into `data/custom/<dataset_name>/`.
    * Scans folder structures (`class_name/image.png`), assigns incremental class IDs, and saves JSON class mappings.
    * `load_dataset_arrays()`: Preprocesses images to $(32 \times 32)$ RGB float32 arrays ready for training.
* **Interconnections**: Powers the "Custom Dataset & Training Engine" in `app.py`.
