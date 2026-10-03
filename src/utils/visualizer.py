"""
Advanced visualizer and HUD renderer for traffic sign detections.
Renders stylized bounding boxes, category-colored badges, and real-time telemetry overlays.
"""

from typing import List, Dict, Tuple
import cv2
import numpy as np

from src.schema import PipelineDetection
from src.gtsrb_classes import get_category_color_bgr


class Visualizer:
    """
    Renders bounding boxes, prediction labels, and telemetry stats onto OpenCV frames.
    """

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.55,
        font_thickness: int = 1,
        show_telemetry: bool = True
    ):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.show_telemetry = show_telemetry
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[PipelineDetection],
        latency_ms: Dict[str, float] = None,
        fps: float = 0.0,
        active_speed_limit: Optional[str] = None,
        active_hazard: Optional[str] = None
    ) -> np.ndarray:
        """
        Draws all detection boxes, labels, and HUD overlays onto a copy of the frame.
        """
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()

        # 1. Draw each detected sign
        for det in detections:
            self._draw_detection(canvas, det)

        # 2. Draw telemetry HUD (FPS, Latency)
        if self.show_telemetry:
            self._draw_telemetry_hud(canvas, fps, latency_ms, len(detections))

        # 3. Draw ADAS Cockpit HUD (Speed Limit Sign & Hazard Alert)
        self._draw_adas_cockpit(canvas, active_speed_limit, active_hazard)

        return canvas

    def _draw_adas_cockpit(
        self,
        canvas: np.ndarray,
        active_speed_limit: Optional[str] = None,
        active_hazard: Optional[str] = None
    ):
        h, w = canvas.shape[:2]

        # A. Active Speed Limit Sign & Braking Advisory at Top-Right
        if active_speed_limit:
            cx, cy, r = w - 60, 58, 38
            # White circular disc with red outer border (European speed limit sign)
            cv2.circle(canvas, (cx, cy), r + 2, (15, 15, 15), -1, cv2.LINE_AA)  # shadow
            cv2.circle(canvas, (cx, cy), r, (35, 35, 235), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), r - 7, (250, 250, 250), -1, cv2.LINE_AA)
            # Text inside
            speed_text = active_speed_limit.replace(" km/h", "").replace("km/h", "").strip()
            (tw, th), _ = cv2.getTextSize(speed_text, self.font, 0.85, 2)
            cv2.putText(canvas, speed_text, (cx - tw // 2, cy + th // 2), self.font, 0.85, (20, 20, 20), 2, cv2.LINE_AA)

            # Calculate theoretical dry stopping distance (Reaction 1.0s + Braking mu=0.7)
            try:
                v = float("".join(filter(str.isdigit, speed_text)))
                stop_dist = int(round((v / 3.6) * 1.0 + (v ** 2) / (250.0 * 0.7)))
                advisory_text = f"STOP DIST: ~{stop_dist}m"
            except Exception:
                advisory_text = "SPEED LIMIT"

            cv2.putText(canvas, advisory_text, (cx - 48, cy + r + 16), self.font, 0.36, (240, 240, 240), 1, cv2.LINE_AA)

        # B. Hazard Alert Banner at Top-Center
        if active_hazard:
            banner_text = f"WARNING: {active_hazard.upper()} AHEAD"
            (tw, th), _ = cv2.getTextSize(banner_text, self.font, 0.55, 2)
            bx1 = (w - tw) // 2 - 18
            by1 = 12
            bx2 = bx1 + tw + 36
            by2 = by1 + th + 20
            # High-visibility warning banner
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (18, 110, 235), cv2.FILLED)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, banner_text, (bx1 + 18, by2 - 7), self.font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_detection(self, canvas: np.ndarray, item: PipelineDetection):
        bbox = item.detection.bbox
        cls_result = item.classification
        is_ambiguous = getattr(cls_result, "is_ambiguous", False)

        # Ambiguous detections colored in caution amber
        color = (0, 215, 255) if is_ambiguous else get_category_color_bgr(cls_result.category)

        x1, y1, x2, y2 = bbox.to_xyxy()

        # Draw main bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.box_thickness, cv2.LINE_AA)

        # Draw corner accents for high-tech HUD look
        corner_len = max(8, min(16, (x2 - x1) // 4, (y2 - y1) // 4))
        c_thick = self.box_thickness + 1
        # Top-left
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, c_thick, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, c_thick, cv2.LINE_AA)
        # Top-right
        cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), color, c_thick, cv2.LINE_AA)
        cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), color, c_thick, cv2.LINE_AA)
        # Bottom-left
        cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), color, c_thick, cv2.LINE_AA)
        cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), color, c_thick, cv2.LINE_AA)
        # Bottom-right
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, c_thick, cv2.LINE_AA)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, c_thick, cv2.LINE_AA)

        # Build label text: "[ID: 14] Stop (96%)" or "[Ambiguous] [ID: 14] Stop"
        conf_pct = int(cls_result.confidence * 100)
        track_prefix = f"{item.detection.detector_label} " if "Track" in item.detection.detector_label else ""
        amb_prefix = "[?] " if is_ambiguous else ""
        label_text = f"{amb_prefix}{track_prefix}[{cls_result.class_id}] {cls_result.class_name} ({conf_pct}%)"

        (tw, th), baseline = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)

        # Determine label badge position (above box, or below if close to top edge)
        badge_y1 = y1 - th - 8
        badge_y2 = y1
        if badge_y1 < 0:
            badge_y1 = y2
            badge_y2 = y2 + th + 8

        badge_x1 = max(0, x1)
        badge_x2 = min(canvas.shape[1], x1 + tw + 10)

        # Draw filled background badge with dark contrast
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), (25, 25, 25), cv2.FILLED)
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, cv2.LINE_AA)

        # Text baseline
        text_y = badge_y2 - 4 if badge_y1 < y1 else badge_y2 - 4
        cv2.putText(
            canvas,
            label_text,
            (badge_x1 + 5, text_y),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness,
            cv2.LINE_AA
        )

        # Draw secondary TT100K consensus tag
        if getattr(item, "tt100k_result", None) is not None:
            tt = item.tt100k_result
            tt_status = "CONSENSUS" if tt.is_consensus else "DISCORD"
            tt_color = (60, 220, 60) if tt.is_consensus else (40, 140, 255)
            tt_text = f"TT100K: {tt.class_code} ({int(tt.confidence * 100)}%) [{tt_status}]"

            (tt_w, tt_h), _ = cv2.getTextSize(tt_text, self.font, self.font_scale * 0.82, self.font_thickness)
            tt_y1 = y2 + 3
            tt_y2 = y2 + tt_h + 9
            if tt_y2 > canvas.shape[0]:
                tt_y1 = max(0, badge_y1 - tt_h - 6)
                tt_y2 = badge_y1 - 2

            tt_x1 = max(0, x1)
            tt_x2 = min(canvas.shape[1], x1 + tt_w + 10)

            cv2.rectangle(canvas, (tt_x1, tt_y1), (tt_x2, tt_y2), (20, 20, 20), cv2.FILLED)
            cv2.rectangle(canvas, (tt_x1, tt_y1), (tt_x2, tt_y2), tt_color, 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                tt_text,
                (tt_x1 + 5, tt_y2 - 3),
                self.font,
                self.font_scale * 0.82,
                (240, 240, 240),
                self.font_thickness,
                cv2.LINE_AA
            )

    def _draw_telemetry_hud(
        self,
        canvas: np.ndarray,
        fps: float,
        latency_ms: Dict[str, float] = None,
        sign_count: int = 0
    ):
        h, w = canvas.shape[:2]
        tt_lat = latency_ms.get("tt100k_ms") if latency_ms else None
        hud_w, hud_h = 240, (92 if tt_lat is not None else 75)

        # Draw semi-transparent HUD background banner at top-left
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), (80, 80, 80), 1, cv2.LINE_AA)

        # Draw telemetry lines
        fps_text = f"FPS: {fps:.1f}" if fps > 0 else "FPS: --"
        total_lat = latency_ms.get("total_ms", 0.0) if latency_ms else 0.0
        det_lat = latency_ms.get("detect_ms", 0.0) if latency_ms else 0.0
        cls_lat = latency_ms.get("classify_ms", 0.0) if latency_ms else 0.0

        cv2.putText(canvas, f"TRAFFIC SIGN PIPELINE", (20, 28), self.font, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{fps_text} | Total: {total_lat:.1f}ms", (20, 48), self.font, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Det: {det_lat:.1f}ms | Cls: {cls_lat:.1f}ms | Signs: {sign_count}", (20, 68), self.font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        if tt_lat is not None:
            cv2.putText(canvas, f"TT100K: {tt_lat:.1f}ms (Consensus Engine)", (20, 85), self.font, 0.38, (120, 230, 120), 1, cv2.LINE_AA)
