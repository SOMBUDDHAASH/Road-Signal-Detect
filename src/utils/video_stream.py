"""Video & Camera Streaming Utilities."""

import time
import cv2
import numpy as np


class CameraStream:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        return self

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
