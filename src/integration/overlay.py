"""ADAS Cockpit HUD & Visual Annotation Engine (Member D).
Renders category-coded bounding boxes, translucent label banners, and an active speed limit HUD.
"""

from typing import List, Optional
import cv2
import numpy as np
from src.schema import IntegratedDetection, SignCategory

CATEGORY_COLORS = {
    SignCategory.PROHIBITORY: (0, 0, 230),     # Bright Red (BGR)
    SignCategory.DANGER: (0, 165, 255),          # Amber / Orange (BGR)
    SignCategory.MANDATORY: (230, 120, 20),      # Blue (BGR)
    SignCategory.OTHER: (50, 205, 50),           # Emerald Green (BGR)
}


def draw_speed_limit_badge(canvas: np.ndarray, speed_limit: int, x: int = 40, y: int = 40, radius: int = 30):
    """Draw a German/European circular speed limit sign onto the cockpit HUD."""
    # Outer Red Border
    cv2.circle(canvas, (x, y), radius, (0, 0, 220), -1)
    # Inner White Disc
    cv2.circle(canvas, (x, y), radius - 6, (255, 255, 255), -1)
    # Speed Number Text
    text = str(speed_limit)
    font_scale = 0.7 if len(text) <= 2 else 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = x - tw // 2
    ty = y + th // 2
    cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def render_cockpit_hud(
    frame: np.ndarray,
    detections: List[IntegratedDetection],
    active_speed_limit: Optional[int] = None,
    fps: float = 0.0,
    latency_ms: float = 0.0,
) -> np.ndarray:
    """Render full integrated HUD overlay on frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Draw each detected sign
    for det in detections:
        box = det.detection.bbox
        x1, y1, x2, y2 = box.to_xyxy()

        cat = det.classification.category if det.classification else SignCategory.OTHER
        color = CATEGORY_COLORS.get(cat, (0, 255, 0))

        # Main bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label text
        track_str = f"#{det.track_id} " if det.track_id else ""
        label = f"{track_str}{det.final_class_name} ({det.final_confidence*100:.1f}%)"

        font_scale = max(0.4, min(w, h) / 1200)
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Translucent pill background
        pad = 4
        banner_y1 = max(0, y1 - th - 2 * pad)
        banner_y2 = y1
        banner_x2 = min(w, x1 + tw + 2 * pad)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, banner_y1), (banner_x2, banner_y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Indicator color pip
        cv2.rectangle(canvas, (x1, banner_y1), (x1 + 4, banner_y2), color, -1)

        # Label text
        cv2.putText(
            canvas,
            label,
            (x1 + 7, banner_y2 - pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # 2. Cockpit Top Banner
    hud_overlay = canvas.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.addWeighted(hud_overlay, 0.65, canvas, 0.35, 0, canvas)

    # Telemetry metrics
    stat_text = f"SIGNS: {len(detections)}  |  LATENCY: {latency_ms:.1f} ms"
    if fps > 0:
        stat_text += f"  |  FPS: {fps:.1f}"

    cv2.putText(canvas, stat_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # 3. Active Speed Limit Badge at top-right
    if active_speed_limit is not None:
        draw_speed_limit_badge(canvas, active_speed_limit, x=w - 50, y=28, radius=22)
        cv2.putText(canvas, "ZONE", (w - 100, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

    return canvas
