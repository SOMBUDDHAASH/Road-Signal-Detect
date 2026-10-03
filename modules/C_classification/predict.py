"""
Direct Prediction Module for Member C (Classification Lead).
Exposes the clean predict_sign(image_path_or_array) interface requested by Member C:
  from predict import predict_sign
  predicted_class, confidence = predict_sign(image_path)
Supports both TensorFlow/Keras (.keras) and PyTorch (.pt) models.
"""

from typing import Union, Tuple
import os
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.classification.model import load_classifier, PyTorchClassifier
from src.gtsrb_classes import get_class_name

# Global cached classifier instance for rapid repeated predictions
_GLOBAL_CLASSIFIER = None


def get_default_classifier():
    """Initializes or returns the active classifier (.keras or .pt)."""
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None:
        # Check if Member C's TensorFlow .keras model is present
        keras_path = os.path.join(PROJECT_ROOT, "models", "traffic_sign_model.keras")
        alt_keras_path = os.path.join(PROJECT_ROOT, "weights", "classification", "traffic_sign_model.keras")

        if os.path.exists(keras_path):
            _GLOBAL_CLASSIFIER = load_classifier(keras_path, color_mode="bgr")
        elif os.path.exists(alt_keras_path):
            _GLOBAL_CLASSIFIER = load_classifier(alt_keras_path, color_mode="bgr")
        else:
            # Fall back to PyTorch classifier
            pt_path = os.path.join(PROJECT_ROOT, "weights", "classification", "classifier.pt")
            _GLOBAL_CLASSIFIER = PyTorchClassifier(model_path=pt_path, auto_fallback=True)

    return _GLOBAL_CLASSIFIER


def predict_sign(image_input: Union[str, np.ndarray]) -> Tuple[int, float]:
    """
    Predict the traffic sign class and confidence for an image path or BGR image crop.

    Args:
      image_input: Either an image file path string or a numpy BGR image array.

    Returns:
      Tuple of (predicted_class_id: int, confidence: float)
    """
    classifier = get_default_classifier()

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        # Read image using OpenCV BGR format (as Member C trained it)
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Could not read image at path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError(f"Expected str path or np.ndarray, got {type(image_input)}")

    result = classifier.classify(image)
    return result.class_id, result.confidence


def main():
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Interactive mode matching Member C's terminal prompt
        img_path = input("Enter the path of a traffic sign image: ").strip()

    if not img_path:
        print("No image path provided.")
        return

    try:
        class_id, confidence = predict_sign(img_path)
        cname = get_class_name(class_id)
        print(f"Prediction: Predicted Class: {class_id} ({cname}) Confidence: {confidence * 100:.1f} %")
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
