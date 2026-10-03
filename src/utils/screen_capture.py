"""
Real-time Screen Capture Handler for continuous detection from screen video playback.
Allows the system to watch and detect traffic signs from any video playing on the laptop!
"""

from typing import Optional, Tuple, Generator
import time
import numpy as np
import cv2
import mss


class ScreenCaptureHandler:
    """
    Captures live screen frames from any window or monitor region using ultra-fast mss.
    """

    def __init__(self, monitor_index: int = 1, region: Optional[Tuple[int, int, int, int]] = None):
        """
        region: (left, top, width, height) in pixels. If None, captures entire primary monitor.
        """
        self.sct = mss.mss()
        self.monitors = self.sct.monitors

        if region:
            left, top, width, height = region
            self.capture_box = {"left": left, "top": top, "width": width, "height": height}
        else:
            # Default to primary monitor (index 1)
            target_idx = min(monitor_index, len(self.monitors) - 1)
            self.capture_box = self.monitors[target_idx]

    def capture_frame(self) -> np.ndarray:
        """Grabs a single screen frame as BGR numpy array."""
        sct_img = self.sct.grab(self.capture_box)
        # Convert BGRA to BGR
        frame = np.array(sct_img)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def frames(self, target_fps: int = 30) -> Generator[np.ndarray, None, None]:
        frame_interval = 1.0 / target_fps
        while True:
            t0 = time.perf_counter()
            frame = self.capture_frame()
            yield frame
            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def close(self):
        if self.sct:
            self.sct.close()
