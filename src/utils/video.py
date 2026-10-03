"""
Video and webcam streaming utilities for the Traffic Sign Detection pipeline.
Supports offline video file inference with annotated video export and live webcam streaming.
"""

from typing import Generator, Optional, Tuple, Callable
import time
import cv2
import numpy as np


class VideoStreamHandler:
    """
    Manages frame ingestion and export for video files and live webcam capture.
    """

    def __init__(self, source: str | int = 0):
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> cv2.VideoCapture:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")
        return self.cap

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    @property
    def properties(self) -> dict:
        if not self.cap:
            self.open()
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0),
            "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def frames(self) -> Generator[np.ndarray, None, None]:
        if not self.cap:
            self.open()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            yield frame

    def process_and_save(
        self,
        output_path: str,
        process_fn: Callable[[np.ndarray], np.ndarray],
        max_frames: Optional[int] = None
    ):
        """
        Processes each frame using process_fn and saves annotated output to video file.
        """
        props = self.properties
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path, fourcc, props["fps"], (props["width"], props["height"])
        )

        count = 0
        try:
            for frame in self.frames():
                annotated = process_fn(frame)
                writer.write(annotated)
                count += 1
                if max_frames and count >= max_frames:
                    break
        finally:
            writer.release()
            self.close()
