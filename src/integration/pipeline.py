"""Master Integration Pipeline (Member D).
Orchestrates:
  1. Input Video / Image Ingestion
  2. Environmental Pre-conditioning (CLAHE)
  3. Sign Localization (Member B - YOLOv8)
  4. Precise Crop Extraction & Padding
  5. Sign Classification (Member C - CNN)
  6. Multi-frame Tracking & Speed Limit Memory
  7. ADAS Cockpit HUD Rendering
  8. Trip Event Telemetry Logging
"""

import time
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import (
    DEFAULT_DET_CONF,
    DEFAULT_DET_IOU,
    DEFAULT_CLS_CONF,
    DEFAULT_BBOX_PADDING,
    CLASS_NAMES,
)
from src.schema import (
    DetectionResult,
    ClassificationResult,
    IntegratedDetection,
    PipelineFrameResult,
)
from src.detection.yolo_detector import YOLODetector
from src.detection.shape_detector import ShapeContourDetector
from src.detection.preprocess import enhance_contrast
from src.classification.keras_classifier import KerasSignClassifier
from src.classification.pytorch_classifier import PyTorchSignClassifier
from src.integration.tracker import SignTracker
from src.integration.overlay import render_cockpit_hud
from src.integration.telemetry import TelemetryLogger


class TrafficSignPipeline:
    def __init__(
        self,
        detector_path: Optional[str] = None,
        classifier_path: Optional[str] = None,
        use_shape_fallback: bool = True,
        classifier_backend: str = "keras",  # "keras" or "pytorch"
    ):
        """Initialize the end-to-end perception pipeline."""
        print("\n=== Initializing Master Perception Pipeline (Member D) ===")

        # 1. Initialize Detector
        try:
            self.detector = YOLODetector(detector_path)
            self.detector_type = "YOLOv8"
        except Exception as e:
            print(f"[PIPELINE WARNING] YOLO initialization failed ({e}). Using ShapeContourDetector fallback.")
            self.detector = ShapeContourDetector()
            self.detector_type = "ShapeContour"

        self.shape_detector = ShapeContourDetector() if use_shape_fallback else None

        # 2. Initialize Classifier
        self.classifier_type = classifier_backend.lower()
        if self.classifier_type == "pytorch":
            try:
                self.classifier = PyTorchSignClassifier(classifier_path)
            except Exception as e:
                print(f"[PIPELINE WARNING] PyTorch classifier failed ({e}). Falling back to Keras.")
                self.classifier = KerasSignClassifier(classifier_path)
                self.classifier_type = "keras"
        else:
            try:
                self.classifier = KerasSignClassifier(classifier_path)
            except Exception as e:
                print(f"[PIPELINE WARNING] Keras classifier failed ({e}). Falling back to PyTorch.")
                self.classifier = PyTorchSignClassifier(classifier_path)
                self.classifier_type = "pytorch"

        # 3. Initialize Tracking & Telemetry
        self.tracker = SignTracker()
        self.telemetry = TelemetryLogger()
        self.frame_counter = 0
        print("[PIPELINE] Ready for inference.\n")

    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = DEFAULT_DET_CONF,
        iou_threshold: float = DEFAULT_DET_IOU,
        cls_threshold: float = DEFAULT_CLS_CONF,
        apply_clahe: bool = False,
        render_overlay: bool = True,
    ) -> PipelineFrameResult:
        """Process a single frame through the entire perception stack."""
        start_t = time.perf_counter()
        self.frame_counter += 1

        if frame is None or frame.size == 0:
            return PipelineFrameResult(
                frame_index=self.frame_counter,
                original_shape=(0, 0, 0),
                detections=[],
                inference_time_ms=0.0,
            )

        # 1. Environmental Pre-conditioning (optional CLAHE)
        processed_input = enhance_contrast(frame) if apply_clahe else frame

        # 2. Stage 1: Detection
        raw_detections: List[DetectionResult] = self.detector.detect(
            processed_input, conf_threshold=conf_threshold, iou_threshold=iou_threshold
        )

        # Fallback to shape detector if YOLO found nothing and shape fallback is enabled
        if not raw_detections and self.shape_detector is not None:
            raw_detections = self.shape_detector.detect(processed_input, conf_threshold=conf_threshold)

        # 3. Stage 2: Cropping & Classification
        integrated_dets: List[IntegratedDetection] = []
        h, w = frame.shape[:2]

        for det in raw_detections:
            crop = det.bbox.crop(frame, padding=DEFAULT_BBOX_PADDING)
            cls_result: Optional[ClassificationResult] = None

            if crop is not None and crop.size > 0:
                try:
                    cls_result = self.classifier.classify(crop)
                except Exception as e:
                    cls_result = None

            # Determine final identity
            if cls_result and cls_result.confidence >= cls_threshold:
                final_cid = cls_result.class_id
                final_cname = cls_result.class_name
                final_conf = cls_result.confidence
            else:
                # If classifier is below threshold or unavailable, use detector's prediction if valid
                if det.class_id >= 0 and det.class_id in CLASS_NAMES:
                    final_cid = det.class_id
                    final_cname = det.class_name
                    final_conf = det.confidence
                elif cls_result:
                    final_cid = cls_result.class_id
                    final_cname = cls_result.class_name
                    final_conf = cls_result.confidence
                else:
                    final_cid = 0
                    final_cname = "Traffic Sign"
                    final_conf = det.confidence

            item = IntegratedDetection(
                detection=det,
                classification=cls_result,
                final_class_id=final_cid,
                final_class_name=final_cname,
                final_confidence=final_conf,
            )
            integrated_dets.append(item)

        # 4. Stage 3: Temporal Tracking & Speed Memory
        tracked_dets = self.tracker.update(integrated_dets, self.frame_counter)

        # 5. Log Telemetry
        for td in tracked_dets:
            spd = td.classification.speed_limit if td.classification else None
            self.telemetry.log_detection(
                frame_idx=self.frame_counter,
                track_id=td.track_id or 0,
                class_id=td.final_class_id,
                class_name=td.final_class_name,
                confidence=td.final_confidence,
                speed_limit=spd,
            )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0

        # 6. Stage 4: Cockpit HUD Overlay
        annotated = None
        if render_overlay:
            annotated = render_cockpit_hud(
                frame=frame,
                detections=tracked_dets,
                active_speed_limit=self.tracker.current_speed_limit,
                fps=fps,
                latency_ms=elapsed_ms,
            )

        return PipelineFrameResult(
            frame_index=self.frame_counter,
            original_shape=frame.shape,
            detections=tracked_dets,
            active_speed_limit=self.tracker.current_speed_limit,
            inference_time_ms=round(elapsed_ms, 2),
            annotated_frame=annotated,
        )

    def process_image(
        self,
        image_input: Union[str, np.ndarray],
        output_path: Optional[str] = None,
        conf_threshold: float = DEFAULT_DET_CONF,
    ) -> PipelineFrameResult:
        """Process a single image path or numpy array."""
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not read image: {image_input}")
        else:
            img = image_input

        result = self.process_frame(img, conf_threshold=conf_threshold, render_overlay=True)

        if output_path and result.annotated_frame is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), result.annotated_frame)
            print(f"[PIPELINE] Annotated image written to: {out_p}")

        return result

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_window: bool = False,
        conf_threshold: float = DEFAULT_DET_CONF,
        progress_callback=None,
    ) -> int:
        """Process a video file frame-by-frame and optionally save the annotated result."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w, h))

        total_detections = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            result = self.process_frame(frame, conf_threshold=conf_threshold, render_overlay=True)
            total_detections += len(result.detections)

            if writer and result.annotated_frame is not None:
                writer.write(result.annotated_frame)

            if progress_callback:
                progress_callback(frame_idx, total_frames, len(result.detections))

            if show_window and result.annotated_frame is not None:
                cv2.imshow("Traffic Sign Detection & Recognition Pipeline", result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
            print(f"[PIPELINE] Video successfully saved to: {output_path}")
        if show_window:
            cv2.destroyAllWindows()

        return total_detections
