"""
Dedicated Road Sign OCR & Character Recognition Engine.
Localizes and recognizes numbers (speed limits, weights, distances) and
alphabetic strings (STOP, YIELD, ZONE, ONE WAY, EXIT, BUS, TAXI, P)
from traffic sign regions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


@dataclass
class SignTextResult:
    raw_string: str
    detected_number: Optional[int]
    detected_word: Optional[str]
    confidence: float
    sign_type: str  # "SPEED_LIMIT", "STOP", "YIELD", "PARKING", "SERVICE", "GENERAL_TEXT"
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) within crop


class RoadSignOCREngine:
    """
    Self-contained OCR and text recognition engine tailored specifically
    for traffic signs, speed limits, and highway alphanumeric markings.
    Operates without requiring external binaries like tesseract.exe.
    """

    KNOWN_SPEEDS = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    KNOWN_WORDS = ["STOP", "YIELD", "ZONE", "ONE WAY", "EXIT", "BUS", "TAXI", "P", "END", "PED", "SLOW"]

    def __init__(self, template_size: Tuple[int, int] = (40, 28)):
        self.template_size = template_size
        self._digit_templates: Dict[str, np.ndarray] = {}
        self._word_templates: Dict[str, np.ndarray] = {}
        self._build_synthetic_templates()

        # Check for optional deep OCR packages
        self._has_easyocr = False
        self._easyocr_reader = None
        try:
            import easyocr
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self._has_easyocr = True
        except Exception:
            pass

    def _tight_crop_binary(self, binary: np.ndarray) -> np.ndarray:
        """Tightly crop a binary template image to its non-zero glyph bounding box."""
        pts = cv2.findNonZero(binary)
        if pts is None:
            return binary
        x, y, w, h = cv2.boundingRect(pts)
        if w > 0 and h > 0:
            return binary[y:y+h, x:x+w].copy()
        return binary

    def _build_synthetic_templates(self):
        """Generate canonical binary templates for digits 0-9 and key traffic words."""
        h, w = self.template_size

        # Generate digit templates '0'-'9'
        for digit in "0123456789":
            canvas = Image.new("L", (w, h), color=255)
            draw = ImageDraw.Draw(canvas)
            draw.text((w // 4, 2), digit, fill=0)
            arr = np.array(canvas, dtype=np.uint8)
            _, binary = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY_INV)
            self._digit_templates[digit] = self._tight_crop_binary(binary)

        # Generate common speed limit compound templates
        for speed in self.KNOWN_SPEEDS:
            s_str = str(speed)
            w_compound = w * len(s_str)
            canvas = Image.new("L", (w_compound, h), color=255)
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 2), s_str, fill=0)
            arr = np.array(canvas, dtype=np.uint8)
            _, binary = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY_INV)
            self._digit_templates[f"speed_{speed}"] = self._tight_crop_binary(binary)

        # Generate key word templates
        for word in ["STOP", "YIELD", "ZONE", "BUS", "TAXI", "P", "EXIT", "END"]:
            w_w = max(40, len(word) * 18)
            canvas = Image.new("L", (w_w, h), color=255)
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 2), word, fill=0)
            arr = np.array(canvas, dtype=np.uint8)
            _, binary = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY_INV)
            self._word_templates[word] = self._tight_crop_binary(binary)

    def preprocess_sign_interior(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract the central region of the sign (ignoring outer borders)
        and compute a clean binary glyph mask.
        """
        h, w = crop.shape[:2]
        if h < 16 or w < 16:
            return np.zeros((h, w), dtype=np.uint8)

        # Center ROI (avoid border ring/triangle)
        pad_y = int(h * 0.18)
        pad_x = int(w * 0.18)
        interior = crop[pad_y:h-pad_y, pad_x:w-pad_x]
        if interior.size == 0:
            interior = crop

        gray = cv2.cvtColor(interior, cv2.COLOR_BGR2GRAY) if len(interior.shape) == 3 else interior.copy()

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        contrast = clahe.apply(gray)

        # Otsu thresholding
        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Invert if the background is dark (e.g. white text on blue/red background)
        # We want text/glyphs to be foreground (255)
        total_pixels = binary.shape[0] * binary.shape[1]
        if cv2.countNonZero(binary) > (total_pixels * 0.55):
            binary = cv2.bitwise_not(binary)

        # Clear connected components touching the crop boundaries (outer sign border rings/triangles)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        ih, iw = binary.shape[:2]
        clean_bin = np.zeros_like(binary)
        for i in range(1, num_labels):
            bx, by, bw, bh, area = stats[i]
            # Skip if touching the outer boundary lines
            if bx <= 1 or by <= 1 or (bx + bw) >= (iw - 1) or (by + bh) >= (ih - 1):
                continue
            clean_bin[labels == i] = 255

        # If clearing removed everything (e.g. inverted text filled whole frame), fall back to original binary
        if cv2.countNonZero(clean_bin) > 10:
            return clean_bin

        return binary

    def detect(self, crop: np.ndarray) -> Optional[SignTextResult]:
        """
        Analyze a traffic sign crop to detect numerical values or strings.
        Returns a structured SignTextResult or None if no text is present.
        """
        if crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return None

        # 1. First check if EasyOCR is available for high-fidelity text
        if self._has_easyocr and self._easyocr_reader:
            try:
                results = self._easyocr_reader.readtext(crop)
                for bbox, text, score in results:
                    text_clean = text.strip().upper()
                    if score > 0.40 and len(text_clean) > 0:
                        # Check if digits
                        num_match = re.search(r'\b(\d{2,3})\b', text_clean)
                        if num_match:
                            val = int(num_match.group(1))
                            if val in self.KNOWN_SPEEDS:
                                return SignTextResult(
                                    raw_string=text_clean,
                                    detected_number=val,
                                    detected_word=None,
                                    confidence=float(score),
                                    sign_type="SPEED_LIMIT"
                                )
                        # Check words
                        for kw in self.KNOWN_WORDS:
                            if kw in text_clean:
                                return SignTextResult(
                                    raw_string=text_clean,
                                    detected_number=None,
                                    detected_word=kw,
                                    confidence=float(score),
                                    sign_type="STOP" if kw == "STOP" else "GENERAL_TEXT"
                                )
            except Exception:
                pass

        # 2. Standalone Built-in Topological & Template OCR Engine
        binary = self.preprocess_sign_interior(crop)
        if binary is None or cv2.countNonZero(binary) < 25:
            return None

        candidates = []

        # Check for speed limit numbers (e.g. 20, 30, 50, 60, 70, 80, 100, 120)
        speed_res = self._match_speed_limits(binary)
        if speed_res:
            candidates.append(speed_res)

        # Check for key keywords (STOP, YIELD, ZONE, P)
        word_res = self._match_keywords(crop, binary)
        if word_res:
            candidates.append(word_res)

        # Check individual digit blobs
        digit_res = self._detect_digit_blobs(binary)
        if digit_res:
            candidates.append(digit_res)

        if candidates:
            candidates.sort(key=lambda c: c.confidence, reverse=True)
            return candidates[0]

        return None

    def _match_speed_limits(self, binary_interior: np.ndarray) -> Optional[SignTextResult]:
        """Match speed limit numerals against compound multi-scale templates."""
        bh, bw = binary_interior.shape[:2]
        if bh < 15 or bw < 15:
            return None

        # Calculate aspect ratio of actual foreground glyphs
        nonzero_pts = cv2.findNonZero(binary_interior)
        if nonzero_pts is None:
            return None
        _, _, gw, gh = cv2.boundingRect(nonzero_pts)
        if gh < 8 or gw < 6:
            return None
        glyph_aspect = gw / float(gh)

        best_speed = None
        best_score = 0.0

        for speed in self.KNOWN_SPEEDS:
            tmpl_key = f"speed_{speed}"
            if tmpl_key not in self._digit_templates:
                continue

            tmpl = self._digit_templates[tmpl_key]
            th, tw = tmpl.shape[:2]
            tmpl_aspect = tw / float(th)

            # Aspect ratio gate: candidate glyph aspect must roughly match template aspect
            # (e.g. 3-digit '100' is ~1.8, 2-digit '50' is ~1.1)
            aspect_diff = abs(glyph_aspect - tmpl_aspect)
            if aspect_diff > 0.55:
                continue

            for scale in [0.85, 0.95, 1.0, 1.05, 1.15]:
                scaled_h = int(round(gh * scale))
                scaled_w = int(round(tw * (scaled_h / float(th))))
                if scaled_h < 4 or scaled_w < 4 or scaled_h >= bh or scaled_w >= bw:
                    continue

                scaled_tmpl = cv2.resize(tmpl, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                res = cv2.matchTemplate(binary_interior, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                # Penalize aspect difference slightly
                adjusted_val = max_val - (aspect_diff * 0.10)
                if adjusted_val > best_score:
                    best_score = adjusted_val
                    best_speed = speed

        # Acceptance threshold for speed limit compound template
        if best_speed is not None and best_score >= 0.45:
            return SignTextResult(
                raw_string=str(best_speed),
                detected_number=best_speed,
                detected_word=None,
                confidence=round(float(best_score), 3),
                sign_type="SPEED_LIMIT"
            )

        return None

    def _match_keywords(self, crop: np.ndarray, binary_interior: np.ndarray) -> Optional[SignTextResult]:
        """Detect standard traffic keywords like STOP, YIELD, ZONE, P."""
        bh, bw = binary_interior.shape[:2]

        nonzero_pts = cv2.findNonZero(binary_interior)
        if nonzero_pts is None:
            return None
        _, _, gw, gh = cv2.boundingRect(nonzero_pts)
        if gh < 8 or gw < 6:
            return None
        glyph_aspect = gw / float(gh)

        best_word = None
        best_score = 0.0

        for word, tmpl in self._word_templates.items():
            th, tw = tmpl.shape[:2]
            tmpl_aspect = tw / float(th)

            # Skip single letter templates when word is multi-letter
            if len(word) == 1 and glyph_aspect > 1.1:
                continue

            # Aspect ratio gate prevents 1-letter templates (like 'P') from matching 4-letter words ('STOP')
            aspect_diff = abs(glyph_aspect - tmpl_aspect)
            if aspect_diff > 0.60:
                continue

            for scale in [0.85, 0.95, 1.0, 1.05, 1.15]:
                scaled_h = int(round(gh * scale))
                scaled_w = int(round(tw * (scaled_h / float(th))))
                if scaled_h < 4 or scaled_w < 4 or scaled_h >= bh or scaled_w >= bw:
                    continue

                scaled_tmpl = cv2.resize(tmpl, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                res = cv2.matchTemplate(binary_interior, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                adjusted_val = max_val - (aspect_diff * 0.10)
                if adjusted_val > best_score:
                    best_score = adjusted_val
                    best_word = word

        if best_word and best_score >= 0.48:
            stype = "STOP" if best_word == "STOP" else ("PARKING" if best_word == "P" else "GENERAL_TEXT")
            return SignTextResult(
                raw_string=best_word,
                detected_number=None,
                detected_word=best_word,
                confidence=round(float(best_score), 3),
                sign_type=stype
            )

        return None

    def _detect_digit_blobs(self, binary_interior: np.ndarray) -> Optional[SignTextResult]:
        """Extract individual character/digit contours and recognize using topology + templates."""
        contours, hierarchy = cv2.findContours(binary_interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bh, bw = binary_interior.shape[:2]

        char_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter character-like blobs: height between 25% and 85% of interior height
            if h >= (bh * 0.22) and h <= (bh * 0.90) and w >= 4 and w <= (bw * 0.60):
                aspect = w / float(h)
                if 0.20 <= aspect <= 1.4:
                    char_candidates.append((x, y, w, h, cnt))

        if not char_candidates:
            return None

        # Sort characters left-to-right
        char_candidates.sort(key=lambda c: c[0])
        recognized_chars = []
        confidences = []

        for x, y, w, h, cnt in char_candidates:
            char_patch = binary_interior[y:y+h, x:x+w]
            char, conf = self._classify_single_digit(char_patch)
            if char:
                recognized_chars.append(char)
                confidences.append(conf)

        if recognized_chars:
            extracted_str = "".join(recognized_chars)
            # If all digits
            if extracted_str.isdigit() and len(extracted_str) in [2, 3]:
                val = int(extracted_str)
                # Map close speed limits (e.g. 50, 60, 30, 80)
                mean_conf = float(np.mean(confidences))
                return SignTextResult(
                    raw_string=extracted_str,
                    detected_number=val,
                    detected_word=None,
                    confidence=round(mean_conf, 3),
                    sign_type="SPEED_LIMIT" if val in self.KNOWN_SPEEDS else "ALPHANUMERIC"
                )

        return None

    def _classify_single_digit(self, char_patch: np.ndarray) -> Tuple[Optional[str], float]:
        """Classify a single extracted binary character patch using topological Euler holes and correlation."""
        h, w = char_patch.shape[:2]
        if h < 8 or w < 3:
            return None, 0.0

        # Standardize patch size
        canonical = cv2.resize(char_patch, (self.template_size[1], self.template_size[0]), interpolation=cv2.INTER_NEAREST)

        # Topological Euler holes count
        # Invert canonical: holes are 0s inside 255 character
        padded = np.pad(canonical, 2, mode='constant', constant_values=0)
        holes_contours, _ = cv2.findContours(cv2.bitwise_not(padded), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Background is 1 contour, each hole is an additional inner contour
        num_holes = max(0, len(holes_contours) - 2)

        best_digit = None
        best_score = -1.0

        # Narrow down candidates by topological hole count
        if num_holes >= 2:
            candidate_digits = ["8", "B"]
        elif num_holes == 1:
            candidate_digits = ["0", "4", "6", "9", "P", "D", "A"]
        else:
            candidate_digits = ["1", "2", "3", "5", "7", "T", "E", "S"]

        for d in candidate_digits:
            if d not in self._digit_templates:
                continue
            tmpl = self._digit_templates[d]
            res = cv2.matchTemplate(canonical, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res[0][0])
            if score > best_score:
                best_score = score
                best_digit = d

        if best_digit and best_score >= 0.42:
            return best_digit, best_score

        return None, 0.0


if __name__ == "__main__":
    ocr = RoadSignOCREngine()

    # Test synthetic Stop sign
    stop_sign = np.zeros((120, 120, 3), dtype=np.uint8)
    stop_sign[:] = (30, 30, 200)  # Red background
    pil_img = Image.fromarray(stop_sign)
    draw = ImageDraw.Draw(pil_img)
    draw.text((25, 45), "STOP", fill=(255, 255, 255))
    test_stop = np.array(pil_img)

    res = ocr.detect(test_stop)
    print("OCR Test Result on Synthetic STOP:", res)
