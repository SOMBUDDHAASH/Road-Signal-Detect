"""
Dedicated Road Sign OCR & Character Recognition Engine.
Localizes and recognizes numbers (speed limits, weights, distances) and
alphabetic strings (STOP, YIELD, ZONE, MPH, KM/H, NO PARKING, EXIT, BUS, TAXI, P)
from traffic sign regions.
Maintained by Member D (Integration & Pipeline Lead).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw


@dataclass
class SignTextResult:
    raw_string: str
    detected_number: Optional[int]
    detected_word: Optional[str]
    confidence: float
    sign_type: str  # "SPEED_LIMIT", "STOP", "YIELD", "PARKING", "NO_PARKING", "SERVICE", "GENERAL_TEXT"
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    is_advisory_speed: bool = False


class RoadSignOCREngine:
    """
    Self-contained OCR and text recognition engine tailored specifically
    for traffic signs, speed limits, and highway alphanumeric markings.
    Combines compound multi-scale template matching with component-level
    character isolation and normalized invariant IoU correlation.
    """

    KNOWN_SPEEDS = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 120, 130]
    KNOWN_WORDS = ["STOP", "YIELD", "ZONE", "MPH", "KM/H", "NO PARKING", "ONE WAY", "EXIT", "BUS", "TAXI", "P", "END", "PED", "SLOW"]

    def __init__(self, char_size: Tuple[int, int] = (32, 32)):
        self.char_size = char_size
        self._digit_templates: Dict[str, np.ndarray] = {}
        self._letter_templates: Dict[str, np.ndarray] = {}
        self._compound_speed_templates: Dict[int, np.ndarray] = {}
        self._compound_word_templates: Dict[str, np.ndarray] = {}
        self._build_templates()

        # Check for optional deep OCR packages
        self._has_easyocr = False
        self._easyocr_reader = None
        try:
            import easyocr
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self._has_easyocr = True
        except Exception:
            pass

    def _build_templates(self):
        """Generate standardized individual and compound binary templates."""
        h, w = self.char_size

        # 1. Individual digits '0'-'9'
        for d in "0123456789":
            canvas = Image.new("L", (40, 50), color=255)
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 4), d, fill=0)
            arr = np.array(canvas, dtype=np.uint8)
            _, bin_d = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY_INV)
            pts = cv2.findNonZero(bin_d)
            if pts is not None:
                bx, by, bw, bh = cv2.boundingRect(pts)
                crop_t = bin_d[by:by+bh, bx:bx+bw]
                self._digit_templates[d] = cv2.resize(crop_t, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Individual letters 'A'-'Z'
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            canvas = Image.new("L", (40, 50), color=255)
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 4), ch, fill=0)
            arr = np.array(canvas, dtype=np.uint8)
            _, bin_d = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY_INV)
            pts = cv2.findNonZero(bin_d)
            if pts is not None:
                bx, by, bw, bh = cv2.boundingRect(pts)
                crop_t = bin_d[by:by+bh, bx:bx+bw]
                self._letter_templates[ch] = cv2.resize(crop_t, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3. Compound Speed Limit Templates (20, 30, 40, 50, 60, 70, 80, 100, 120, etc.)
        for s in self.KNOWN_SPEEDS:
            s_str = str(s)
            canvas = Image.new("L", (55 if s >= 100 else 40, 28), color=255)
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 2), s_str, fill=0)
            arr = np.array(canvas, dtype=np.uint8)
            _, bin_d = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY_INV)
            pts = cv2.findNonZero(bin_d)
            if pts is not None:
                bx, by, bw, bh = cv2.boundingRect(pts)
                self._compound_speed_templates[s] = bin_d[by:by+bh, bx:bx+bw]

        # 4. Compound Word Templates (STOP, ZONE, YIELD, MPH, NO PARKING, etc.)
        for word in self.KNOWN_WORDS:
            canvas = Image.new("L", (max(40, len(word) * 16), 28), color=255)
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 2), word, fill=0)
            arr = np.array(canvas, dtype=np.uint8)
            _, bin_d = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY_INV)
            pts = cv2.findNonZero(bin_d)
            if pts is not None:
                bx, by, bw, bh = cv2.boundingRect(pts)
                self._compound_word_templates[word] = bin_d[by:by+bh, bx:bx+bw]

    def _match_char_iou(self, char_mask: np.ndarray, template_dict: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """Compute IoU between a character binary mask and standardized templates."""
        char_resized = (cv2.resize(char_mask, self.char_size, interpolation=cv2.INTER_NEAREST) > 0)
        best_char = "?"
        best_iou = -1.0

        for ch, tmpl in template_dict.items():
            tmpl_bin = (tmpl > 0)
            intersection = np.logical_and(char_resized, tmpl_bin).sum()
            union = np.logical_or(char_resized, tmpl_bin).sum()
            iou = intersection / float(max(1, union))
            if iou > best_iou:
                best_iou = iou
                best_char = ch

        return best_char, float(best_iou)

    def extract_line_components(self, binary: np.ndarray) -> List[List[Tuple[int, int, int, int]]]:
        """
        Locates characters and groups them into horizontal text lines.
        Excludes border rings, large geometric contours, and tiny speckles.
        """
        h, w = binary.shape[:2]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        candidates = []
        for i in range(1, num_labels):
            bx, by, bw, bh = stats[i, :4]
            area = stats[i, cv2.CC_STAT_AREA]
            aspect = bw / float(max(1, bh))

            if area < 8 or area > (w * h * 0.35):
                continue
            if bw > (w * 0.65) or bh > (h * 0.65):
                continue
            if aspect < 0.15 or aspect > 2.5:
                continue
            if bx <= 2 or by <= 2 or (bx + bw) >= (w - 2) or (by + bh) >= (h - 2):
                continue

            candidates.append((bx, by, bw, bh))

        if not candidates:
            return []

        candidates.sort(key=lambda b: (b[1], b[0]))

        lines: List[List[Tuple[int, int, int, int]]] = []
        for box in candidates:
            bx, by, bw, bh = box
            center_y = by + (bh / 2.0)
            placed = False
            for line in lines:
                ref_box = line[0]
                ref_center_y = ref_box[1] + (ref_box[3] / 2.0)
                ref_h = ref_box[3]
                if abs(center_y - ref_center_y) < max(12, ref_h * 0.50):
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])

        for line in lines:
            line.sort(key=lambda b: b[0])

        return lines

    def detect(self, crop: np.ndarray) -> Optional[SignTextResult]:
        """
        Analyzes a sign crop across multiple binarization channels to recognize
        speed limit numerals or regulatory words (STOP, YIELD, MPH, etc.).
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            return None

        h, w = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Multi-channel masks
        # 1. Dark text on light background (e.g. speed limit white disk, yellow diamond)
        _, dark_on_light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Light text on dark background (e.g. STOP, Blue arrows/roundabout, white text on gray)
        white_on_dark = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 80, 255]))
        _, light_gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        combined_light = cv2.bitwise_or(white_on_dark, light_gray)

        # Check dominant yellow background for advisory speed
        yellow_px = cv2.countNonZero(cv2.inRange(hsv, np.array([15, 65, 45]), np.array([35, 255, 255])))
        is_yellow = (yellow_px / float(h * w)) >= 0.28

        # --- Phase 1: High-Speed Compound Template Matching ---
        # 1A. Test compound speed templates (20, 30, 50, 70, 100, etc.) on dark_on_light mask
        for speed, tmpl in self._compound_speed_templates.items():
            th, tw = tmpl.shape[:2]
            if th >= h or tw >= w:
                continue
            res = cv2.matchTemplate(dark_on_light, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val >= 0.85:
                return SignTextResult(
                    raw_string=str(speed),
                    detected_number=speed,
                    detected_word=None,
                    confidence=round(float(max_val), 3),
                    sign_type="SPEED_LIMIT",
                    is_advisory_speed=is_yellow
                )

        # 1B. Test compound word templates (STOP, ZONE, YIELD) on combined_light mask
        for word in ["STOP", "ZONE", "YIELD", "MPH", "NO PARKING"]:
            if word in self._compound_word_templates:
                tmpl = self._compound_word_templates[word]
                th, tw = tmpl.shape[:2]
                if th >= h or tw >= w:
                    continue
                res = cv2.matchTemplate(combined_light, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val >= 0.85:
                    stype = "STOP" if word == "STOP" else ("YIELD" if word == "YIELD" else "GENERAL_TEXT")
                    return SignTextResult(
                        raw_string=word,
                        detected_number=None,
                        detected_word=word,
                        confidence=round(float(max_val), 3),
                        sign_type=stype
                    )

        # --- Phase 2: Component-Level Line & Character IoU Extraction ---
        # 2A. Check White-on-Dark channel for STOP
        white_lines = self.extract_line_components(white_on_dark)
        for line in white_lines:
            if len(line) == 4:
                chars = []
                ious = []
                for (bx, by, bw, bh) in line:
                    char_mask = white_on_dark[by:by+bh, bx:bx+bw]
                    ch, iou = self._match_char_iou(char_mask, self._letter_templates)
                    chars.append(ch)
                    ious.append(iou)
                word = "".join(chars)
                avg_iou = float(np.mean(ious))
                if word == "STOP" or ("S" in word and "T" in word and "P" in word):
                    return SignTextResult(
                        raw_string="STOP",
                        detected_number=None,
                        detected_word="STOP",
                        confidence=max(0.92, avg_iou),
                        sign_type="STOP"
                    )

        # 2B. Check Dark-on-Light channel (Speed limits, MPH, general text)
        dark_lines = self.extract_line_components(dark_on_light)
        detected_number: Optional[int] = None
        detected_suffix: Optional[str] = None
        best_num_iou = 0.0

        for line in dark_lines:
            if 1 <= len(line) <= 3:
                num_str = ""
                line_ious = []
                for (bx, by, bw, bh) in line:
                    char_mask = dark_on_light[by:by+bh, bx:bx+bw]
                    d, iou = self._match_char_iou(char_mask, self._digit_templates)
                    num_str += d
                    line_ious.append(iou)
                if num_str.isdigit() and len(line_ious) > 0:
                    val = int(num_str)
                    if val in self.KNOWN_SPEEDS:
                        detected_number = val
                        best_num_iou = float(np.mean(line_ious))

            if len(line) >= 2:
                word_str = ""
                for (bx, by, bw, bh) in line:
                    char_mask = dark_on_light[by:by+bh, bx:bx+bw]
                    ch, _ = self._match_char_iou(char_mask, self._letter_templates)
                    word_str += ch
                if "MPH" in word_str or "MP" in word_str:
                    detected_suffix = "MPH"
                elif "KM" in word_str:
                    detected_suffix = "km/h"

        if detected_number is not None:
            raw_s = f"{detected_number} {detected_suffix}" if detected_suffix else str(detected_number)
            return SignTextResult(
                raw_string=raw_s,
                detected_number=detected_number,
                detected_word=detected_suffix,
                confidence=round(max(0.75, best_num_iou), 3),
                sign_type="SPEED_LIMIT",
                is_advisory_speed=is_yellow or (detected_suffix == "MPH")
            )

        return None


# Global singleton OCR engine
_GLOBAL_OCR_ENGINE: Optional[RoadSignOCREngine] = None

def get_ocr_engine() -> RoadSignOCREngine:
    global _GLOBAL_OCR_ENGINE
    if _GLOBAL_OCR_ENGINE is None:
        _GLOBAL_OCR_ENGINE = RoadSignOCREngine()
    return _GLOBAL_OCR_ENGINE
