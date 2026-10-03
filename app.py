"""
Streamlit Web Dashboard for Road / Traffic Sign Detection & Recognition.
Japanese Minimalist Aesthetic (Kanso 簡素, Shibui 渋味, Ma 間).

Features:
  1. Test Against Established Samples (GTSRB Benchmark - All 43 Classes)
  2. Test Against Still Images (File Upload)
  3. Continuous Real-Time Video Testing via Camera / Webcam (Zero False Body Detections)
  4. Continuous Video Testing via Screenshare (High-Speed Screen Capture via mss)
  5. Test Against YouTube Video Links (via yt-dlp)
  - Live Local Timestamped Event Stream (e.g. 03/10/2026 16:20:05 -> Stop sign detected)
  - Hot-Swappable YOLO & PyTorch Classifier Models
  - Custom Dataset Uploader & On-the-Fly PyTorch Trainer
Maintained by Member D (Integration & Pipeline Lead).
"""

from typing import List, Dict, Optional, Tuple
import os
import glob
import time
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import TrafficSignPipeline
from src.gtsrb_classes import GTSRB_CLASSES, GTSRB_CATEGORIES, get_sign_category
from src.schema import SignCategory, PipelineResult, DetectionResult, BoundingBox, PipelineDetection
from src.detection.yolo import YOLODetector
from src.detection.shape_detector import RobustTrafficSignDetector
from src.classification.model import PyTorchClassifier
from src.classification.mock import ColorHeuristicClassifier
from src.utils.screen_capture import ScreenCaptureHandler
from src.utils.youtube import YouTubeStreamHandler
from src.utils.event_logger import DetectionEventLogger
from src.dataset.custom_dataset import CustomDatasetManager
from src.dataset.benchmark_loader import BenchmarkDataLoader

# Page Configuration
st.set_page_config(
    page_title="Traffic Sign Recognition • Kanso Edition",
    page_icon="⛩️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Japanese Minimalist Aesthetic Styling (Kanso, Shibui, Ma)
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1A1C1E;
        letter-spacing: -0.01em;
    }

    /* Minimalist Zen Header */
    .zen-header-container {
        padding: 1.2rem 0 1.6rem 0;
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        margin-bottom: 1.8rem;
    }
    .zen-kanji-sub {
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #8C939D;
        margin-bottom: 0.35rem;
    }
    .zen-title {
        font-size: 2.15rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #111315;
        line-height: 1.15;
        margin: 0;
    }
    .zen-lead {
        font-size: 0.95rem;
        color: #5A626A;
        margin-top: 0.45rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* Minimalist Status Pills */
    .zen-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        border: 1px solid rgba(0, 0, 0, 0.08);
        background: #F8F9FA;
        color: #3B4045;
    }
    .zen-pill-active {
        background: #F0FDF4;
        color: #166534;
        border-color: #BBF7D0;
    }
    .zen-pill-accent {
        background: #FEF2F2;
        color: #991B1B;
        border-color: #FECACA;
    }
    .zen-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: currentColor;
    }

    /* Subtle Card Containers */
    .zen-card {
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .zen-card-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 0.8rem;
    }

    /* Minimalist Category Badges */
    .badge-prohibitory {
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.02em;
    }
    .badge-danger {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.02em;
    }
    .badge-mandatory {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.02em;
    }
    .badge-other {
        background-color: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.02em;
    }

    /* Minimalist Monospace Event Log */
    .zen-log-terminal {
        background-color: #111315;
        color: #E2E8F0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        max-height: 220px;
        overflow-y: auto;
        border: 1px solid rgba(255, 255, 255, 0.08);
        line-height: 1.7;
    }
    .zen-log-entry {
        display: flex;
        gap: 12px;
        align-items: baseline;
    }
    .zen-log-time {
        color: #94A3B8;
        font-size: 0.78rem;
    }
    .zen-log-text {
        color: #38BDF8;
    }

    /* Telemetry Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #111315 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #64748B !important;
        font-weight: 600 !important;
    }

    /* Clean Dividers */
    hr {
        margin: 1.5rem 0 !important;
        border: none !important;
        border-top: 1px solid rgba(0, 0, 0, 0.08) !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "event_logger" not in st.session_state:
    st.session_state.event_logger = DetectionEventLogger(dedup_cooldown_sec=2.5)

if "custom_yolo_path" not in st.session_state:
    st.session_state.custom_yolo_path = None

if "custom_classifier_path" not in st.session_state:
    st.session_state.custom_classifier_path = None


def get_active_pipeline(
    detector_choice: str,
    classifier_choice: str,
    padding_ratio: float = 0.05,
    custom_yolo_path: Optional[str] = None,
    custom_cls_path: Optional[str] = None
) -> TrafficSignPipeline:
    """Builds the active pipeline with hot-swappable detector and classifier models."""
    # Detector selection
    if detector_choice == "Custom Uploaded YOLO" and custom_yolo_path and os.path.exists(custom_yolo_path):
        detector = YOLODetector(model_path=custom_yolo_path, auto_fallback=True)
    elif detector_choice.startswith("Fine-tuned Traffic YOLO"):
        detector = YOLODetector(model_path=os.path.join("weights", "detection", "best.pt"), auto_fallback=True)
    elif detector_choice.startswith("Robust Contour"):
        detector = RobustTrafficSignDetector()
    else:
        detector = YOLODetector(model_path=os.path.join("weights", "detection", "best.pt"), auto_fallback=True)

    # Classifier selection
    if classifier_choice == "Custom Uploaded Classifier" and custom_cls_path and os.path.exists(custom_cls_path):
        classifier = PyTorchClassifier(model_path=custom_cls_path, auto_fallback=True)
    elif classifier_choice.startswith("PyTorch GTSRB CNN"):
        classifier = PyTorchClassifier(model_path=os.path.join("weights", "classification", "classifier.pt"), auto_fallback=True)
    elif classifier_choice.startswith("Color Heuristic"):
        classifier = ColorHeuristicClassifier()
    else:
        classifier = PyTorchClassifier(model_path=os.path.join("weights", "classification", "classifier.pt"), auto_fallback=True)

    return TrafficSignPipeline(
        detector=detector,
        classifier=classifier,
        crop_padding_ratio=padding_ratio,
        enable_tracking=True
    )


def render_event_log_ui(event_logger: DetectionEventLogger):
    """Renders the minimalist monospace event log with local timestamps."""
    st.markdown("#### ⏱️ Real-Time Telemetry Stream (Local Timecode)")
    recent_logs = event_logger.get_recent_logs(limit=25)

    if recent_logs:
        log_entries = []
        for line in recent_logs:
            if "->" in line:
                parts = line.split("->", 1)
                log_entries.append(f"<div class='zen-log-entry'><span class='zen-log-time'>{parts[0].strip()}</span> <span class='zen-log-text'>→ {parts[1].strip()}</span></div>")
            else:
                log_entries.append(f"<div class='zen-log-entry'><span class='zen-log-text'>{line}</span></div>")

        log_html = "<div class='zen-log-terminal'>" + "".join(log_entries) + "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

        c_csv, c_json, c_clr = st.columns([1, 1, 4])
        with c_csv:
            st.download_button(
                "📥 Export CSV",
                data=event_logger.export_csv(),
                file_name=f"detections_{int(time.time())}.csv",
                mime="text/csv"
            )
        with c_json:
            st.download_button(
                "📥 Export JSON",
                data=event_logger.export_json(),
                file_name=f"detections_{int(time.time())}.json",
                mime="application/json"
            )
        with c_clr:
            if st.button("🗑️ Clear Log"):
                event_logger.clear()
                st.rerun()
    else:
        st.caption("No events logged yet. Detections will appear here in real-time with local timecode.")


def get_badge_html(category: str) -> str:
    cat_lower = category.lower()
    if "prohibitory" in cat_lower:
        return f"<span class='badge-prohibitory'>{category}</span>"
    elif "danger" in cat_lower:
        return f"<span class='badge-danger'>{category}</span>"
    elif "mandatory" in cat_lower:
        return f"<span class='badge-mandatory'>{category}</span>"
    else:
        return f"<span class='badge-other'>{category}</span>"


def main():
    # Zen Header
    st.markdown("""
    <div class="zen-header-container">
        <div class="zen-kanji-sub">交通標識識別 • Traffic Sign Recognition System</div>
        <h1 class="zen-title">Integrated ADAS Vision Pipeline</h1>
        <div class="zen-lead">German Traffic Sign Recognition Benchmark (GTSRB 43 Classes) • Modular Integration Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Model & Weight Engine (Hot-Swappable)
    st.sidebar.markdown("### ⛩️ Model & Weights Engine")

    # 1. Detector Switcher
    detector_options = [
        "Fine-tuned Traffic YOLO (best.pt)",
        "Robust Contour & Shape Detector",
        "Custom Uploaded YOLO"
    ]
    chosen_detector = st.sidebar.selectbox("Active Localization Model", detector_options, index=0)

    if chosen_detector == "Custom Uploaded YOLO":
        uploaded_yolo = st.sidebar.file_uploader("Upload YOLO Model (.pt)", type=["pt", "onnx"], key="yolo_uploader")
        if uploaded_yolo:
            save_path = os.path.join("weights", "detection", "custom_uploaded_yolo.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_yolo.getbuffer())
            st.session_state.custom_yolo_path = save_path
            st.sidebar.success("✅ Custom YOLO Model Activated!")

    # 2. Classifier Switcher
    classifier_options = [
        "PyTorch GTSRB CNN (classifier.pt - 99.2% Acc)",
        "Color Heuristic Baseline",
        "Custom Uploaded Classifier"
    ]
    chosen_classifier = st.sidebar.selectbox("Active Classifier Model", classifier_options, index=0)

    if chosen_classifier == "Custom Uploaded Classifier":
        uploaded_cls = st.sidebar.file_uploader("Upload Classifier (.pt)", type=["pt", "onnx"], key="cls_uploader")
        if uploaded_cls:
            save_path = os.path.join("weights", "classification", "custom_uploaded_classifier.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_cls.getbuffer())
            st.session_state.custom_classifier_path = save_path
            st.sidebar.success("✅ Custom Classifier Activated!")

    conf_thresh = st.sidebar.slider("Confidence Gate", min_value=0.10, max_value=0.95, value=0.35, step=0.05)
    padding_ratio = st.sidebar.slider("Crop Margin Ratio", min_value=0.0, max_value=0.20, value=0.05, step=0.02)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 System Architecture")
    st.sidebar.markdown("""
    <div style="display:flex; flex-direction:column; gap:6px;">
        <span class="zen-pill zen-pill-active"><span class="zen-dot"></span> Pipeline: Standalone Ready</span>
        <span class="zen-pill zen-pill-active"><span class="zen-dot"></span> Member A: Data Pipeline</span>
        <span class="zen-pill zen-pill-active"><span class="zen-dot"></span> Member B: YOLOv8 Detector</span>
        <span class="zen-pill zen-pill-active"><span class="zen-dot"></span> Member C: GTSRB 43 Classifier</span>
        <span class="zen-pill zen-pill-active"><span class="zen-dot"></span> Member D: Master Integration</span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Navigation")
    app_mode = st.sidebar.radio(
        "Select Testing Mode:",
        [
            "1. GTSRB Benchmark Explorer (43 Classes)",
            "2. Still Image Testing (Upload)",
            "3. Live Video via Camera (Webcam / Dashcam)",
            "4. Continuous Screen Capture (Laptop Video)",
            "5. YouTube Dashcam Stream (yt-dlp)",
            "📁 Custom Dataset & Training Engine"
        ]
    )

    # Initialize Active Pipeline
    pipeline = get_active_pipeline(
        detector_choice=chosen_detector,
        classifier_choice=chosen_classifier,
        padding_ratio=padding_ratio,
        custom_yolo_path=st.session_state.custom_yolo_path,
        custom_cls_path=st.session_state.custom_classifier_path
    )
    logger = st.session_state.event_logger

    # =========================================================================
    # OPTION 1: GTSRB BENCHMARK EXPLORER (ALL 43 CLASSES)
    # =========================================================================
    if app_mode.startswith("1."):
        st.subheader("📊 Option 1: GTSRB Benchmark Explorer")
        st.markdown(
            "Evaluate any of the **43 official German Traffic Sign classes** against the active model. "
            "Includes ground-truth ROI coordinates, reference icons from `Meta.csv`, and exact classification accuracy metrics."
        )

        bench_loader = BenchmarkDataLoader("data")
        sample_dir = os.path.join("data", "samples")
        sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.png")))

        if not sample_files:
            st.warning("No sample files found in data/samples. Extracting...")
            return

        c_filter1, c_filter2 = st.columns([1, 2])
        with c_filter1:
            category_filter = st.selectbox(
                "Filter by Sign Category",
                ["All Categories (43 Classes)", "Prohibitory (Speed Limits)", "Danger / Warning", "Mandatory", "Other / Priority"]
            )

        # Filter available samples based on category
        available_files = []
        for f in sample_files:
            bname = os.path.basename(f)
            gt = bench_loader.get_ground_truth(bname)
            if not gt:
                available_files.append(bname)
                continue
            cat = gt.category.lower()
            if category_filter.startswith("Prohibitory") and "prohibitory" not in cat:
                continue
            elif category_filter.startswith("Danger") and "danger" not in cat:
                continue
            elif category_filter.startswith("Mandatory") and "mandatory" not in cat:
                continue
            elif category_filter.startswith("Other") and "other" not in cat:
                continue
            available_files.append(bname)

        if not available_files:
            available_files = [os.path.basename(f) for f in sample_files]

        with c_filter2:
            chosen_file = st.selectbox("Select Test Benchmark Sample", available_files)

        img_path = os.path.join(sample_dir, chosen_file)
        frame = cv2.imread(img_path)
        gt = bench_loader.get_ground_truth(chosen_file)

        if frame is not None:
            result = pipeline.process_frame(frame, conf_threshold=conf_thresh, is_video=False)
            logger.log_detections(result.detections)

            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                st.markdown("##### 📌 Reference Icon")
                meta_icon_path = os.path.join("data", "meta", f"{gt.class_id}.png") if gt else None
                if meta_icon_path and os.path.exists(meta_icon_path):
                    st.image(meta_icon_path, caption=f"Class [{gt.class_id}] Icon", width=110)
                else:
                    st.caption("Meta icon loading...")
                if gt:
                    st.markdown(get_badge_html(gt.category), unsafe_allow_html=True)

            with col2:
                st.markdown("##### 📷 Raw Test Image")
                st.image(cv2.cvtColor(result.original_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col3:
                st.markdown("##### 🎯 Model Prediction & HUD")
                st.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Ground Truth vs Prediction Comparison Card
            if gt is not None:
                st.markdown("---")
                pred_id = result.detections[0].classification.class_id if result.detections else -1
                pred_name = result.detections[0].classification.class_name if result.detections else "Unrecognized / Filtered"
                pred_conf = result.detections[0].classification.confidence if result.detections else 0.0

                c_gt1, c_gt2, c_gt3 = st.columns(3)
                with c_gt1:
                    st.markdown(f"**Ground Truth Label**:<br>`[{gt.class_id}] {gt.class_name}`", unsafe_allow_html=True)
                    st.caption(f"Shape ID: `{gt.shape_id}` • Color ID: `{gt.color_id}`")
                with c_gt2:
                    st.markdown(f"**Ground Truth ROI Box**:<br>`({gt.roi_x1}, {gt.roi_y1}, {gt.roi_x2}, {gt.roi_y2})`", unsafe_allow_html=True)
                    st.caption(f"Resolution: {gt.width}x{gt.height} px")
                with c_gt3:
                    st.markdown(f"**Model Prediction**:<br>`[{pred_id}] {pred_name}`", unsafe_allow_html=True)
                    st.caption(f"Confidence: **{pred_conf*100:.1f}%**")

                if pred_id == gt.class_id:
                    st.success(f"✅ **EXACT BENCHMARK MATCH**: Model correctly recognized `{gt.class_name}` ({pred_conf*100:.1f}% confidence)!")
                elif pred_id >= 0:
                    st.warning(f"⚠️ Predicted `{pred_name}` vs Ground Truth `{gt.class_name}`.")
                else:
                    st.info("ℹ️ Detection threshold filtered this crop. Adjust the confidence slider in the sidebar.")

            # Telemetry Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Signs Detected", result.num_signs_detected)
            m2.metric("Total Latency", f"{result.total_latency_ms:.1f} ms")
            m3.metric("Localization Time", f"{result.latency_ms['detect_ms']:.1f} ms")
            m4.metric("Classification Time", f"{result.latency_ms['classify_ms']:.1f} ms")

            st.markdown("---")
            render_event_log_ui(logger)

    # =========================================================================
    # OPTION 2: TEST AGAINST STILL IMAGES (UPLOAD)
    # =========================================================================
    elif app_mode.startswith("2."):
        st.subheader("🖼️ Option 2: Test Against Still Images")
        st.markdown("Upload any traffic scene image from your device (`.png`, `.jpg`, `.jpeg`, `.webp`).")

        uploaded = st.file_uploader("Select Image File", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                result = pipeline.process_frame(frame, conf_threshold=conf_thresh, is_video=False)
                logger.log_detections(result.detections)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 📷 Original Image")
                    st.image(cv2.cvtColor(result.original_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                with c2:
                    st.markdown("##### 🎯 Detection & Classification HUD")
                    st.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Signs Detected", result.num_signs_detected)
                m2.metric("Total Latency", f"{result.total_latency_ms:.1f} ms")
                m3.metric("Throughput", f"{result.fps:.1f} FPS")

                st.markdown("---")
                render_event_log_ui(logger)

    # =========================================================================
    # OPTION 3: CONTINUOUS REAL-TIME VIDEO (WEBCAM)
    # =========================================================================
    elif app_mode.startswith("3."):
        st.subheader("📹 Option 3: Continuous Real-Time Video (Webcam / Dashcam)")
        st.markdown("Stream live from a USB webcam or vehicle dashcam with anti-false-detection protection and ADAS HUD telemetry.")

        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            cam_idx = st.number_input("Camera Device Index", min_value=0, max_value=5, value=0)
        with col_c2:
            target_reticle = st.checkbox("🎯 Center Sign Focus Mode (Hold phone or printed sign to camera)", value=True)

        run_cam = st.toggle("▶️ Start Live Camera Stream", value=False)

        if run_cam:
            cap = cv2.VideoCapture(int(cam_idx))
            if not cap.isOpened():
                st.error(f"Could not open camera device #{cam_idx}. Check USB/camera permissions.")
            else:
                st_video = st.empty()
                m_cols = st.columns(5)
                m_signs = m_cols[0].empty()
                m_speed = m_cols[1].empty()
                m_det = m_cols[2].empty()
                m_cls = m_cols[3].empty()
                m_fps = m_cols[4].empty()

                while run_cam and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    h, w = frame.shape[:2]

                    # Process raw frame through pipeline first
                    result = pipeline.process_frame(frame, conf_threshold=conf_thresh)

                    # If Center Reticle is enabled, evaluate center target box safely
                    if target_reticle:
                        box_size = int(min(h, w) * 0.45)
                        cx1 = (w - box_size) // 2
                        cy1 = (h - box_size) // 2
                        cx2 = cx1 + box_size
                        cy2 = cy1 + box_size
                        center_crop = frame[cy1:cy2, cx1:cx2]

                        # Check saturation & skin ratio to distinguish signs from human face / wall
                        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
                        sat = float(np.mean(hsv[:, :, 1]))
                        ycrcb = cv2.cvtColor(center_crop, cv2.COLOR_BGR2YCrCb)
                        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
                        skin_ratio = cv2.countNonZero(skin_mask) / float(center_crop.shape[0] * center_crop.shape[1])

                        is_sign = False
                        if sat >= 45.0 and skin_ratio < 0.22:
                            cls_res = pipeline.classifier.classify(center_crop)
                            if cls_res.class_id >= 0 and cls_res.confidence >= max(0.70, conf_thresh):
                                is_sign = True
                                reticle_color = (0, 255, 0)
                                cv2.rectangle(result.annotated_frame, (cx1, cy1), (cx2, cy2), reticle_color, 2, cv2.LINE_AA)
                                cv2.putText(
                                    result.annotated_frame,
                                    f"HOLD: {cls_res.class_name} ({cls_res.confidence*100:.0f}%)",
                                    (cx1 + 10, cy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, reticle_color, 2, cv2.LINE_AA
                                )
                                p_det = PipelineDetection(
                                    detection=DetectionResult(bbox=BoundingBox(cx1, cy1, cx2, cy2), confidence=cls_res.confidence, detector_label="traffic_sign"),
                                    classification=cls_res,
                                    crop=center_crop
                                )
                                logger.log_detections([p_det])

                        if not is_sign:
                            # Draw subtle grey targeting brackets
                            reticle_color = (180, 180, 180)
                            bracket_len = 25
                            cv2.line(result.annotated_frame, (cx1, cy1), (cx1 + bracket_len, cy1), reticle_color, 2, cv2.LINE_AA)
                            cv2.line(result.annotated_frame, (cx1, cy1), (cx1, cy1 + bracket_len), reticle_color, 2, cv2.LINE_AA)
                            cv2.line(result.annotated_frame, (cx2, cy1), (cx2 - bracket_len, cy1), reticle_color, 2, cv2.LINE_AA)
                            cv2.line(result.annotated_frame, (cx2, cy1), (cx2, cy1 + bracket_len), reticle_color, 2, cv2.LINE_AA)
                            cv2.line(result.annotated_frame, (cx1, cy2), (cx1 + bracket_len, cy2), reticle_color, 2, cv2.LINE_AA)
                            cv2.line(result.annotated_frame, (cx1, cy2), (cx1, cy2 - bracket_len), reticle_color, 2, cv2.LINE_AA)
                            cv2.line(result.annotated_frame, (cx2, cy2), (cx2 - bracket_len, cy2), reticle_color, 2, cv2.LINE_AA)
                            cv2.line(result.annotated_frame, (cx2, cy2), (cx2, cy2 - bracket_len), reticle_color, 2, cv2.LINE_AA)
                            cv2.putText(result.annotated_frame, "HOLD SIGN IN THIS BOX", (cx1 + 10, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

                    logger.log_detections(result.detections)

                    st_video.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    m_signs.metric("Visible Signs", result.num_signs_detected)
                    m_speed.metric("Active Speed", result.active_speed_limit or "None")
                    m_det.metric("Detect Time", f"{result.latency_ms['detect_ms']:.1f} ms")
                    m_cls.metric("Classify Time", f"{result.latency_ms['classify_ms']:.1f} ms")
                    m_fps.metric("Stream FPS", f"{result.fps:.1f} FPS")

                    time.sleep(0.01)

                cap.release()

        st.markdown("---")
        render_event_log_ui(logger)

    # =========================================================================
    # OPTION 4: CONTINUOUS VIDEO (SCREENSHARE / LAPTOP VIDEO)
    # =========================================================================
    elif app_mode.startswith("4."):
        st.subheader("🖥️ Option 4: Continuous Video via Screenshare")
        st.markdown("Captures video playing directly on your laptop screen (browser, media player, dashcam footage) at 30 FPS.")

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            screen_mode = st.radio("Screen Region", ["Full Primary Screen", "Center Video Region (Recommended)"])
        with c_s2:
            st.info("Tip: Open any driving video in your media player or browser, then toggle Start Stream.")

        run_screen = st.toggle("▶️ Start Screen Detection", value=False)
        if run_screen:
            st_screen = st.empty()
            handler = ScreenCaptureHandler()
            m_s_cols = st.columns(4)
            m_s1 = m_s_cols[0].empty()
            m_s2 = m_s_cols[1].empty()
            m_s3 = m_s_cols[2].empty()
            m_s4 = m_s_cols[3].empty()

            custom_bbox = None
            if "Center" in screen_mode:
                mon = handler.get_screen_dimensions()
                mw, mh = mon["width"], mon["height"]
                custom_bbox = {"top": int(mh * 0.15), "left": int(mw * 0.15), "width": int(mw * 0.70), "height": int(mh * 0.70)}

            while run_screen:
                screen_frame = handler.grab_frame(custom_bbox)
                if screen_frame is None:
                    break

                result = pipeline.process_frame(screen_frame, conf_threshold=conf_thresh)
                logger.log_detections(result.detections)

                st_screen.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                m_s1.metric("Visible Signs", result.num_signs_detected)
                m_s2.metric("Active Speed", result.active_speed_limit or "None")
                m_s3.metric("Latency", f"{result.total_latency_ms:.1f} ms")
                m_s4.metric("FPS", f"{result.fps:.1f} FPS")

                time.sleep(0.01)

        st.markdown("---")
        render_event_log_ui(logger)

    # =========================================================================
    # OPTION 5: YOUTUBE DASHCAM STREAMING
    # =========================================================================
    elif app_mode.startswith("5."):
        st.subheader("🌐 Option 5: YouTube Dashcam Stream")
        st.markdown("Paste any YouTube driving video URL to stream and recognize signs continuously via `yt-dlp`.")

        sample_url = "https://www.youtube.com/watch?v=F_fW3WdC1_E"
        yt_url = st.text_input("YouTube Video URL", value=sample_url)

        c_y1, c_y2 = st.columns(2)
        with c_y1:
            stream_mode = st.radio("Processing Mode", ["Direct Progressive Stream", "Download 15s Local Clip"])

        run_yt = st.toggle("▶️ Start YouTube Detection Stream", value=False)
        if run_yt and yt_url:
            yt_handler = YouTubeStreamHandler()
            with st.spinner("Connecting to YouTube stream via yt-dlp..."):
                if "Progressive" in stream_mode:
                    stream_url = yt_handler.get_direct_stream_url(yt_url)
                else:
                    stream_url = yt_handler.download_clip(yt_url, duration_sec=15)

            if not stream_url:
                st.error("Could not resolve stream URL. Ensure yt-dlp has network access.")
            else:
                cap = cv2.VideoCapture(stream_url)
                st_yt = st.empty()
                m_y_cols = st.columns(4)
                my1 = m_y_cols[0].empty()
                my2 = m_y_cols[1].empty()
                my3 = m_y_cols[2].empty()
                my4 = m_y_cols[3].empty()

                while run_yt and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    result = pipeline.process_frame(frame, conf_threshold=conf_thresh)
                    logger.log_detections(result.detections)

                    st_yt.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    my1.metric("Signs Detected", result.num_signs_detected)
                    my2.metric("Active Speed", result.active_speed_limit or "None")
                    my3.metric("Latency", f"{result.total_latency_ms:.1f} ms")
                    my4.metric("Stream FPS", f"{result.fps:.1f} FPS")

                    time.sleep(0.02)

                cap.release()

        st.markdown("---")
        render_event_log_ui(logger)

    # =========================================================================
    # CUSTOM DATASET & TRAINING ENGINE
    # =========================================================================
    elif app_mode.startswith("📁"):
        st.subheader("📁 Custom Dataset Upload & Training Engine")
        st.markdown("Upload regional traffic sign datasets (e.g. Indian, US MUTCD) in ZIP format for dynamic on-the-fly training.")

        custom_manager = CustomDatasetManager()
        zip_file = st.file_uploader("Upload Dataset ZIP (Folder per class: class_name/image.png)", type=["zip"])

        if zip_file:
            with st.spinner("Unpacking and indexing custom dataset..."):
                dataset_name = os.path.splitext(zip_file.name)[0]
                summary = custom_manager.extract_and_index_zip(zip_file.getvalue(), dataset_name)
                st.success(f"Extracted dataset '{summary['dataset_name']}': {summary['total_images']} images across {summary['num_classes']} classes.")
                st.json(summary["class_mapping"])

        datasets = custom_manager.list_datasets()
        if datasets:
            st.markdown("#### Available Custom Datasets")
            selected_ds = st.selectbox("Select Dataset to Train On", datasets)

            c_t1, c_t2, c_t3 = st.columns(3)
            with c_t1:
                epochs = st.number_input("Epochs", min_value=1, max_value=50, value=5)
            with c_t2:
                batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32)
            with c_t3:
                lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.05, value=0.001, format="%.4f")

            if st.button("🚀 Train Custom Model Now"):
                from src.train_custom import train_model
                with st.spinner("Loading images and training PyTorch CNN..."):
                    images, labels, mapping = custom_manager.load_dataset_arrays(selected_ds)
                    if len(images) < 10:
                        st.error("Dataset has too few images to train reliably.")
                    else:
                        out_weights = os.path.join("weights", "classification", f"{selected_ds}_classifier.pt")
                        split = int(len(images) * 0.8)
                        val_acc, out_path = train_model(
                            images[:split], labels[:split],
                            images[split:], labels[split:],
                            num_classes=len(mapping),
                            epochs=int(epochs),
                            batch_size=int(batch_size),
                            learning_rate=float(lr),
                            output_path=out_weights
                        )
                        st.success(f"🎉 Training complete! Validation Accuracy: {val_acc:.2%}. Model saved to {out_path}.")
                        st.session_state.custom_classifier_path = out_path
                        st.info("Custom model automatically hot-swapped into the active pipeline!")


if __name__ == "__main__":
    main()
