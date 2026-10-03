# Module C: GTSRB Traffic Sign Classification (43 Classes)

**Module Owner:** Member C (Sneha Chakraborty)  
**Git Branch:** `feature/classification`  
**Downstream Consumers:** Member D (Integration Pipeline, Temporal Tracker, UI HUD, FastAPI)

---

## 1. Overview & Dual Framework Support (TensorFlow/Keras & PyTorch)

Member D (Integration Lead) has built full **dual-framework support** into the platform:
* **TensorFlow / Keras**: Native support for Member C's `traffic_sign_model.keras` or `.h5` models via `TensorFlowClassifier`.
* **PyTorch / TorchScript**: Support for `.pt` / `.pth` models via `PyTorchClassifier`.

You **do NOT need to rewrite your CNN in PyTorch**. The integration layer automatically recognizes `.keras` models and loads them with your exact training preprocessing:

| Framework | File Format | Adapter Class | Color Space | Input Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **TensorFlow / Keras** | `traffic_sign_model.keras` | `TensorFlowClassifier` | **Native BGR** (OpenCV) | $32 \times 32$, normalized `/ 255.0` |
| **PyTorch** | `classifier.pt` | `PyTorchClassifier` | RGB | $32 \times 32$, normalized `/ 255.0` |

---

## 2. Accuracy Benchmark Metrics

* **Member C Model Performance (TensorFlow CNN)**:
  * Random held-out split: **99.34%**
  * Official GTSRB test set: **95.20%**
  * ROI evaluation: **87.92%**
* **Reference PyTorch Model**: 99.22% validation accuracy.

---

## 3. Important Preprocessing Detail: BGR Channel Ordering

Member C's training script loaded images using:
```python
image = cv2.imread(image_path)
image = cv2.resize(image, (32, 32))
X = X.astype("float32") / 255.0
```
Notice that OpenCV `imread` returns **BGR** format. Member C did **not** convert to RGB during training.
Therefore, `TensorFlowClassifier` uses `color_mode="bgr"` by default, feeding your CNN the exact color channels it learned during training.

---

## 4. How to Drop In Your TensorFlow Model (`.keras`)

### Step 1: Place Your Model File
Copy your trained model to either:
* `models/traffic_sign_model.keras`
* `weights/classification/traffic_sign_model.keras`

*(Or upload it directly through the Streamlit web dashboard in the sidebar under **Model & Weights Engine**).*

### Step 2: Test Single-Image Prediction
Run the interactive prediction script:
```bash
python modules/C_classification/predict.py data/samples/class_14_sample_1.png
```
Or interactively:
```bash
python modules/C_classification/predict.py
# Enter the path of a traffic sign image: data/Train/14/00014_00000_00003.png
# Prediction: Predicted Class: 14 (Stop) Confidence: 100.0 %
```

### Step 3: Use the Direct Python Interface
You or Member D can import the function anywhere:
```python
from modules.C_classification.predict import predict_sign

predicted_class, confidence = predict_sign("data/samples/class_14_sample_1.png")
print(f"Class: {predicted_class}, Conf: {confidence:.2%}")
```

---

## 5. Folder-Based Dataset Loader (`data_loader.py`)

If you want to reload or train directly from class folders (`Train/0/`, `Train/1/`, ... `Train/42/`):
```python
from modules.C_classification.data_loader import load_data

X_train, y_train = load_data("data/Train")
print("Images shape:", X_train.shape)  # (N, 32, 32, 3) in float32
print("Labels shape:", y_train.shape)  # (N,) in int64
```

---

## 6. Verification & Automated Tests

Run the test suite to verify your classifier integration:
```bash
python -m pytest tests/test_member_c_compatibility.py -v
python -m pytest modules/C_classification/test_module_c.py -v
```

Launch the Streamlit dashboard to test live in the browser:
```bash
streamlit run app.py
```
Select **TensorFlow / Keras CNN (traffic_sign_model.keras - Member C)** in the sidebar!

---

## 7. Production Engineering Upgrades (Required for Real Pipeline)

Standard Softmax layers force overconfident guesses on ambiguous or out-of-distribution inputs (e.g. blurry signs triggering 99% false positives). When deploying the production classifier, Member C must implement:

1. **Temperature Scaling**:
   - Scale final logits ($z_i / T$) with $T \approx 1.3$ before applying Softmax.
   - Softens extreme probability distributions without altering the top-1 rank order ($\arg\max_i z_i$).
2. **Epistemic Uncertainty Filtering**:
   - Compute Shannon Entropy $H(p) = -\sum_{i=1}^{K} p_i \log_2(p_i)$ on the output probability distribution.
   - Enforce an entropy ceiling (e.g., $H(p) > 2.3$) to flag or drop ambiguous, damaged, or out-of-distribution inputs, forcing the system to reject random hallucinations rather than guessing.
3. **Hierarchical Loss Training**:
   - Train your CNN with a two-level loss function: a Super-Category loss (differentiating Prohibitory vs Danger vs Mandatory vs Priority signs first) followed by fine-grained class classification.


