import os
import cv2
import numpy as np


# -----------------------------
# 1. Settings
# -----------------------------

IMAGE_SIZE = 32
NUM_CLASSES = 43

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "traffic_sign_model.keras"
)


# -----------------------------
# 2. Load trained model
# -----------------------------

try:
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    model = keras.models.load_model(MODEL_PATH)
except Exception:
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load model via standard loaders: {e}")
        model = None

if model is not None:
    print("Model loaded successfully!")


# -----------------------------
# 3. Prediction function
# -----------------------------

def predict_sign(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image!")
        return None

    # Resize image
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # Make prediction
    prediction = model.predict(
        image,
        verbose=0
    )

    # Get predicted class
    predicted_class = np.argmax(
        prediction[0]
    )

    # Get confidence
    confidence = np.max(
        prediction[0]
    )

    return predicted_class, confidence


# -----------------------------
# 4. Test the model
# -----------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        try:
            image_path = input("Enter the path of a traffic sign image: ")
        except (EOFError, KeyboardInterrupt):
            image_path = "data/samples/class_14_sample_1.png"

    result = predict_sign(image_path)

    if result is not None:
        predicted_class, confidence = result
        print("\nPrediction:")
        print("Predicted Class:", predicted_class)
        print("Confidence:", round(confidence * 100, 2), "%")