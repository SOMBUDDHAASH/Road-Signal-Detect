"""
Unified Traffic Sign Detection & Recognition Pipeline.
Orchestrates detection, crop extraction, classification, and visualization.
Maintained by Member D (Integration & Pipeline Lead).
"""

from typing import List, Optional, Dict, Tuple
import time
import os
import argparse
import cv2
import numpy as np

from src.schema import (
    DetectionResult,
    ClassificationResult,
    PipelineDetection,
    PipelineResult,
    BoundingBox,
    SignCategory
)
from src.detection.base import BaseDetector
from src.detection.mock import MockDetector, ColorContourDetector
from src.classification.base import BaseClassifier
from src.classification.mock import MockClassifier, ColorHeuristicClassifier
from src.utils.visualizer import Visualizer

# Mapping YOLO detector classes to standard GTSRB classes for fusion
YOLO_LABEL_TO_GTSRB: Dict[str, Tuple[int, str, SignCategory]] = {
    "Stop": (14, "Stop", SignCategory.PROHIBITORY),
    "Speed Limit 20": (0, "Speed limit (20km/h)", SignCategory.PROHIBITORY),
    "Speed Limit 30": (1, "Speed limit (30km/h)", SignCategory.PROHIBITORY),
    "Speed Limit 50": (2, "Speed limit (50km/h)", SignCategory.PROHIBITORY),
    "Speed Limit 60": (3, "Speed limit (60km/h)", SignCategory.PROHIBITORY),
    "Speed Limit 70": (4, "Speed limit (70km/h)", SignCategory.PROHIBITORY),
    "Speed Limit 80": (5, "Speed limit (80km/h)", SignCategory.PROHIBITORY),
    "Speed Limit 100": (7, "Speed limit (100km/h)", SignCategory.PROHIBITORY),
    "Speed Limit 120": (8, "Speed limit (120km/h)", SignCategory.PROHIBITORY),
}


from src.tracking.tracker import TemporalSignTracker


class TrafficSignPipeline:
    """
    Master pipeline integrating Member B's detector and Member C's classifier.
    Supports continuous temporal tracking, bounding box smoothing, and vehicle HUD state.
    """

    def __init__(
        self,
        detector: BaseDetector,
        classifier: BaseClassifier,
        visualizer: Optional[Visualizer] = None,
        crop_padding_ratio: float = 0.05,
        default_conf_threshold: float = 0.50,
        enable_tracking: bool = True
    ):
        self.detector = detector
        self.classifier = classifier
        self.visualizer = visualizer or Visualizer()
        self.crop_padding_ratio = crop_padding_ratio
        self.default_conf_threshold = default_conf_threshold
        self.enable_tracking = enable_tracking
        self.tracker = TemporalSignTracker() if enable_tracking else None

        # Smoothing for FPS
        self._prev_frame_time = time.perf_counter()
        self._smoothed_fps = 0.0

    @classmethod
    def create(cls, mode: str = "heuristic", **kwargs) -> "TrafficSignPipeline":
        """
        Factory method to initialize pipeline in different configurations:
          - 'mock': Unit testing without deep learning or computer vision dependencies.
          - 'heuristic': CV color-contour detection + heuristic classifier (immediate live demo).
          - 'production': YOLO detector (Member B) + PyTorch CNN classifier (Member C).
        """
        mode = mode.lower()
        if mode == "mock":
            detector = kwargs.get("detector") or MockDetector()
            classifier = kwargs.get("classifier") or MockClassifier()
        elif mode == "heuristic":
            from src.detection.shape_detector import RobustTrafficSignDetector
            detector = kwargs.get("detector") or RobustTrafficSignDetector()
            classifier = kwargs.get("classifier") or ColorHeuristicClassifier()
        elif mode == "production":
            from src.detection.yolo import YOLODetector
            from src.classification.model import PyTorchClassifier
            detector = kwargs.get("detector") or YOLODetector()
            classifier = kwargs.get("classifier") or PyTorchClassifier()
        else:
            raise ValueError(f"Unknown pipeline mode '{mode}'. Choose 'mock', 'heuristic', or 'production'.")

        return cls(detector=detector, classifier=classifier, **kwargs)

    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        is_video: bool = True
    ) -> PipelineResult:
        """
        Processes a single image frame through the full pipeline:
        1. Localize traffic signs (Detection)
        2. Crop and pad candidate regions
        3. Identify sign classes (Classification) with detector-classifier fusion
        4. Temporal tracking (for video) or direct confirmation (for still images)
        5. Fuse results, calculate latency, and render HUD annotations
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or invalid.")

        conf_threshold = conf_threshold if conf_threshold is not None else self.default_conf_threshold
        h, w = frame.shape[:2]

        # Stage 1: Detection
        t_det_start = time.perf_counter()
        raw_detections: List[DetectionResult] = self.detector.detect(frame, conf_threshold=conf_threshold)

        # If image is already a cropped sign sample (< 150px) and detector found nothing, evaluate whole crop
        if not raw_detections and max(w, h) <= 150:
            raw_detections = [DetectionResult(
                bbox=BoundingBox(0, 0, w, h),
                confidence=1.0,
                detector_label="traffic_sign",
                detector_class_id=0
            )]
        det_latency_ms = (time.perf_counter() - t_det_start) * 1000.0

        # Stage 2: Cropping with safety clamping
        valid_crops: List[np.ndarray] = []
        valid_detections: List[DetectionResult] = []

        for det in raw_detections:
            padded_bbox = det.bbox.pad(self.crop_padding_ratio, max_width=w, max_height=h)
            x1, y1, x2, y2 = padded_bbox.to_xyxy()

            # Guard against zero-area or boundary degenerate boxes
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    valid_crops.append(crop)
                    valid_detections.append(det)

        # Stage 3: Classification & Smart Multi-Stage Fusion
        t_cls_start = time.perf_counter()
        pipeline_detections: List[PipelineDetection] = []

        if valid_crops:
            # Batch classification if supported, else sequential
            cls_results: List[ClassificationResult] = self.classifier.classify_batch(valid_crops)
            for det, cls_res, crop in zip(valid_detections, cls_results, valid_crops):
                # Smart Fusion: If detector already identified a specific sign class with high confidence
                # and classifier is lower confidence or unclassified, trust detector!
                if det.detector_label in YOLO_LABEL_TO_GTSRB and det.confidence >= 0.35:
                    yolo_cid, yolo_name, yolo_cat = YOLO_LABEL_TO_GTSRB[det.detector_label]
                    if cls_res.class_id < 0 or (cls_res.class_id != yolo_cid and cls_res.confidence < det.confidence):
                        cls_res = ClassificationResult(
                            class_id=yolo_cid,
                            class_name=yolo_name,
                            confidence=max(cls_res.confidence, det.confidence),
                            category=yolo_cat
                        )

                # Accept verified traffic sign classes
                if cls_res.class_id >= 0 and cls_res.confidence >= conf_threshold:
                    pipeline_detections.append(PipelineDetection(
                        detection=det,
                        classification=cls_res,
                        crop=crop
                    ))
        cls_latency_ms = (time.perf_counter() - t_cls_start) * 1000.0

        # Stage 4: Temporal Tracking & Anti-Flicker (for Continuous Video/Driving)
        speed_limit = None
        hazard = None
        if is_video and self.tracker:
            pipeline_detections = self.tracker.update(pipeline_detections)
            speed_limit = self.tracker.current_speed_limit
            hazard = self.tracker.active_hazard_warning
        else:
            # For still images, bypass multi-frame tracking and directly extract vehicle state
            speed_map = {
                0: "20 km/h", 1: "30 km/h", 2: "50 km/h", 3: "60 km/h",
                4: "70 km/h", 5: "80 km/h", 7: "100 km/h", 8: "120 km/h"
            }
            for d in pipeline_detections:
                cid = d.classification.class_id
                if cid in speed_map and not speed_limit:
                    speed_limit = speed_map[cid]
                if d.classification.category == SignCategory.DANGER and not hazard:
                    hazard = d.classification.class_name

        # Stage 5: Performance Telemetry & HUD Visualization
        total_latency_ms = det_latency_ms + cls_latency_ms
        current_time = time.perf_counter()
        instant_fps = 1.0 / max(1e-5, current_time - self._prev_frame_time)
        self._prev_frame_time = current_time
        self._smoothed_fps = 0.9 * self._smoothed_fps + 0.1 * instant_fps if self._smoothed_fps > 0 else instant_fps

        latency_dict = {
            "detect_ms": round(det_latency_ms, 2),
            "classify_ms": round(cls_latency_ms, 2),
            "total_ms": round(total_latency_ms, 2),
        }

        annotated = self.visualizer.annotate(
            frame=frame,
            detections=pipeline_detections,
            latency_ms=latency_dict,
            fps=self._smoothed_fps,
            active_speed_limit=speed_limit,
            active_hazard=hazard
        )

        return PipelineResult(
            original_frame=frame,
            annotated_frame=annotated,
            detections=pipeline_detections,
            latency_ms=latency_dict,
            fps=round(self._smoothed_fps, 1),
            active_speed_limit=speed_limit,
            active_hazard=hazard
        )


def main():
    parser = argparse.ArgumentParser(description="Run Traffic Sign Detection & Recognition Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image or video file")
    parser.add_argument("--output", "-o", type=str, default="output_annotated.jpg", help="Path to save annotated output")
    parser.add_argument("--mode", "-m", type=str, default="heuristic", choices=["mock", "heuristic", "production"])
    parser.add_argument("--conf", "-c", type=float, default=0.50, help="Confidence threshold")
    args = parser.parse_args()

    pipeline = TrafficSignPipeline.create(mode=args.mode, default_conf_threshold=args.conf)

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        return

    # Check if input is image or video
    ext = os.path.splitext(args.input)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        image = cv2.imread(args.input)
        if image is None:
            print("Failed to read image.")
            return

        result = pipeline.process_frame(image)
        cv2.imwrite(args.output, result.annotated_frame)
        print(f"[Success] Processed image in {result.total_latency_ms:.1f}ms. Signs detected: {result.num_signs_detected}.")
        print(f"Saved annotated output to: {args.output}")
        for i, det in enumerate(result.detections, 1):
            print(f"  Sign {i}: [{det.classification.class_id}] {det.classification.class_name} "
                  f"(Conf: {det.classification.confidence*100:.1f}%) at {det.detection.bbox.to_xyxy()}")
    else:
        from src.utils.video import VideoStreamHandler
        handler = VideoStreamHandler(args.input)
        print(f"Processing video {args.input} -> {args.output}...")
        handler.process_and_save(
            args.output,
            process_fn=lambda f: pipeline.process_frame(f).annotated_frame
        )
        print(f"[Success] Video processing complete. Saved to: {args.output}")


if __name__ == "__main__":
    main()
