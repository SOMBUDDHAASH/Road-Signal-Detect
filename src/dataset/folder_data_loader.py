"""
GTSRB Folder-Based Dataset Loader for Member C.
Loads images from folder structure: data/Train/<class_id>/*.png
Preserves native OpenCV BGR color ordering and 32x32 normalization.
"""

from modules.C_classification.data_loader import load_data

__all__ = ["load_data"]
