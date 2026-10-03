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
from src.utils.audio_alert import get_audio_transducer
from src.utils.environmental import get_environmental_conditioner

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
    .zen-pill-inactive {
        background: #F8FAFC;
        color: #94A3B8;
        border-color: #E2E8F0;
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
    custom_cls_path: Optional[str] = None,
    enable_plague_detector: bool = True,
    enable_ocr: bool = True,
    enable_stage2_proposals: bool = True,
    enable_semantic_verification: bool = True,
    enable_environmental_enhancer: bool = False,
    enable_tt100k: bool = False
) -> TrafficSignPipeline:
    """Builds the active pipeline with hot-swappable detector and classifier models."""
    # Detector selection
    if detector_choice == "Custom Uploaded YOLO" and custom_yolo_path and os.path.exists(custom_yolo_path):
        detector = YOLODetector(
            model_path=custom_yolo_path,
            auto_fallback=True,
            enable_stage2_proposals=enable_stage2_proposals
        )
    elif detector_choice.startswith("Fine-tuned Traffic YOLO"):
        detector = YOLODetector(
            model_path=os.path.join("weights", "detection", "best.pt"),
            auto_fallback=True,
            enable_stage2_proposals=enable_stage2_proposals
        )
    elif detector_choice.startswith("Robust Contour"):
        detector = RobustTrafficSignDetector()
    else:
        detector = YOLODetector(
            model_path=os.path.join("weights", "detection", "best.pt"),
            auto_fallback=True,
            enable_stage2_proposals=enable_stage2_proposals
        )

    # Classifier selection
    if classifier_choice == "Custom Uploaded Classifier" and custom_cls_path and os.path.exists(custom_cls_path):
        from src.classification.model import load_classifier
        classifier = load_classifier(custom_cls_path, auto_fallback=True)
    elif classifier_choice.startswith("TensorFlow / Keras"):
        from src.classification.tf_classifier import TensorFlowClassifier
        classifier = TensorFlowClassifier(auto_fallback=True)
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
        enable_tracking=True,
        enable_plague_detector=enable_plague_detector,
        enable_ocr=enable_ocr,
        enable_stage2_proposals=enable_stage2_proposals,
        enable_semantic_verification=enable_semantic_verification,
        enable_environmental_enhancer=enable_environmental_enhancer,
        enable_tt100k=enable_tt100k
    )


def render_audio_alerts(result: PipelineResult, enable_audio: bool):
    """Renders non-blocking HTML5 Web Audio chimes and Web Speech voice announcements."""
    if enable_audio and result.detections:
        audio_html = get_audio_transducer().generate_html_audio_payload(
            result.detections,
            enable_sound=True,
            enable_speech=True
        )
        if audio_html:
            st.components.v1.html(audio_html, height=0, width=0)


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
        "TensorFlow / Keras CNN (traffic_sign_model.keras - Member C)",
        "Color Heuristic Baseline",
        "Custom Uploaded Classifier"
    ]
    chosen_classifier = st.sidebar.selectbox("Active Classifier Model", classifier_options, index=0)

    if chosen_classifier == "Custom Uploaded Classifier":
        uploaded_cls = st.sidebar.file_uploader("Upload Classifier (.pt, .onnx, .keras, .h5)", type=["pt", "onnx", "keras", "h5"], key="cls_uploader")
        if uploaded_cls:
            ext = os.path.splitext(uploaded_cls.name)[1].lower()
            save_path = os.path.join("weights", "classification", f"custom_uploaded_classifier{ext}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_cls.getbuffer())
            st.session_state.custom_classifier_path = save_path
            st.sidebar.success(f"✅ Custom Classifier ({ext}) Activated!")

    conf_thresh = st.sidebar.slider("Confidence Gate", min_value=0.10, max_value=0.95, value=0.35, step=0.05)
    padding_ratio = st.sidebar.slider("Crop Margin Ratio", min_value=0.0, max_value=0.20, value=0.05, step=0.02)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Perception Pipeline Controls")

    # Primary GTSRB Deep Learning Callout Card (Always Prioritized)
    st.sidebar.markdown("""
    <div style="background-color: rgba(34, 197, 94, 0.07); border: 1px solid rgba(34, 197, 94, 0.28); border-radius: 8px; padding: 10px 12px; margin-bottom: 12px;">
        <div style="color: #15803d; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Primary System (Always Prioritized)</div>
        <div style="color: #14532d; font-size: 0.85rem; font-weight: 600; margin-top: 3px;">⚡ GTSRB 43-Class CNN + YOLO</div>
        <div style="color: #166534; font-size: 0.74rem; margin-top: 2px;">Primary deep learning models execute first on every input frame.</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<div style='font-size:0.78rem; font-weight:600; text-transform: uppercase; letter-spacing:0.04em; color:#64748B; margin: 10px 0 6px 0;'>Secondary Methods & Fallback Toggles:</div>", unsafe_allow_html=True)

    toggle_plague = st.sidebar.toggle(
        "🦠 Plague Cellular Detector",
        value=True,
        help="Cellular automaton color-pair floodfill fallback. Triggers only when primary GTSRB model finds 0 signs."
    )
    toggle_ocr = st.sidebar.toggle(
        "🔤 Alphanumeric OCR Engine",
        value=True,
        help="Scans detected sign crops for speed limit numerals and highway text indicators."
    )
    toggle_stage2 = st.sidebar.toggle(
        "📐 Stage 2 Shape & Color Proposals",
        value=True,
        help="Geometric shape & color proposals. When OFF, detector operates in 100% strict pure YOLO localization."
    )
    toggle_verifier = st.sidebar.toggle(
        "🛡️ Semantic Consistency Gate",
        value=True,
        help="Verifies detected sign colors & geometries against GTSRB taxonomy to prevent false alarms."
    )
    toggle_audio = st.sidebar.toggle(
        "🔊 ADAS Voice & Audio Alerts",
        value=True,
        help="Synthesizes Web Audio chimes and Web Speech voice announcements for Stop, Speed Limits, and Hazards."
    )
    toggle_env = st.sidebar.toggle(
        "🌧️ Adverse Weather & Night Enhancer",
        value=True,
        help="Applies adaptive CLAHE, low-light gamma correction, and dehazing for rain, fog, and night scenes."
    )
    toggle_tt100k = st.sidebar.toggle(
        "🇨🇳 TT100K Secondary Analysis Model",
        value=False,
        help="Tsinghua-Tencent 100K 221-class deep learning model for cross-domain validation and consensus checking."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 System Architecture")
    plague_badge_cls = "zen-pill-active" if toggle_plague else "zen-pill-inactive"
    ocr_badge_cls = "zen-pill-active" if toggle_ocr else "zen-pill-inactive"
    stage2_badge_cls = "zen-pill-active" if toggle_stage2 else "zen-pill-inactive"
    audio_badge_cls = "zen-pill-active" if toggle_audio else "zen-pill-inactive"
    env_badge_cls = "zen-pill-active" if toggle_env else "zen-pill-inactive"
    tt100k_badge_cls = "zen-pill-active" if toggle_tt100k else "zen-pill-inactive"
    st.sidebar.markdown(f"""
    <div style="display:flex; flex-direction:column; gap:6px;">
        <span class="zen-pill zen-pill-active"><span class="zen-dot"></span> Pipeline: Standalone Ready</span>
        <span class="zen-pill zen-pill-active"><span class="zen-dot"></span> Member A: Data Pipeline</span>
        <span class="zen-pill zen-pill-active"><span class="zen-dot"></span> Member B: YOLOv8 Detector</span>
        <span class="zen-pill zen-pill-active"><span class="zen-dot"></span> Member C: GTSRB 43 Classifier</span>
        <span class="zen-pill zen-pill-active"><span class="zen-dot"></span> Member D: Master Integration</span>
        <span class="zen-pill {audio_badge_cls}"><span class="zen-dot"></span> Transducer: Voice & Audio Alerts</span>
        <span class="zen-pill {env_badge_cls}"><span class="zen-dot"></span> Conditioner: Adverse Weather & Night</span>
        <span class="zen-pill {plague_badge_cls}"><span class="zen-dot"></span> Secondary: Plague Fallback ({'Active' if toggle_plague else 'Bypassed'})</span>
        <span class="zen-pill {ocr_badge_cls}"><span class="zen-dot"></span> Secondary: OCR Engine ({'Active' if toggle_ocr else 'Bypassed'})</span>
        <span class="zen-pill {stage2_badge_cls}"><span class="zen-dot"></span> Proposals: Stage 2 ({'Active' if toggle_stage2 else 'Pure YOLO'})</span>
        <span class="zen-pill {tt100k_badge_cls}"><span class="zen-dot"></span> Secondary: TT100K 221-Class ({'Active' if toggle_tt100k else 'Bypassed'})</span>
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
            "6. 🌐 100 Common Signs & Color Taxonomy Explorer",
            "📁 Custom Dataset & Training Engine"
        ]
    )

    # Initialize Active Pipeline
    pipeline = get_active_pipeline(
        detector_choice=chosen_detector,
        classifier_choice=chosen_classifier,
        padding_ratio=padding_ratio,
        custom_yolo_path=st.session_state.custom_yolo_path,
        custom_cls_path=st.session_state.custom_classifier_path,
        enable_plague_detector=toggle_plague,
        enable_ocr=toggle_ocr,
        enable_stage2_proposals=toggle_stage2,
        enable_semantic_verification=toggle_verifier,
        enable_environmental_enhancer=toggle_env,
        enable_tt100k=toggle_tt100k
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

        # Separate canonical class samples (class_XX_sample_Y.png) and general test samples (00000.png)
        canonical_samples = []
        general_samples = []
        for f in sample_files:
            bname = os.path.basename(f)
            if bname.startswith("class_") and "_sample_" in bname:
                try:
                    parts = bname.split("_")
                    cid = int(parts[1])
                    sidx = int(parts[3].split(".")[0])
                    canonical_samples.append((cid, sidx, bname))
                except Exception:
                    general_samples.append(bname)
            else:
                general_samples.append(bname)

        canonical_samples.sort(key=lambda x: (x[0], x[1]))
        general_samples.sort()

        explorer_mode = st.radio(
            "Benchmark Exploration Mode",
            ["Canonical Benchmark Classes (0 - 42)", "Raw Test Set Slices (Test.csv)", "⚡ Batch Benchmark Evaluation Suite"],
            horizontal=True
        )

        chosen_file = None

        if explorer_mode.startswith("Canonical"):
            c_filter1, c_filter2, c_filter3 = st.columns([1.2, 2.2, 1])
            with c_filter1:
                category_filter = st.selectbox(
                    "Filter by Sign Category",
                    ["All Categories (43 Classes)", "Prohibitory (Speed Limits)", "Danger / Warning", "Mandatory", "Other / Priority"]
                )

            # Filter matching class IDs
            matching_cids = []
            for cid in range(43):
                cat = get_sign_category(cid).value.lower()
                if category_filter.startswith("Prohibitory") and "prohibitory" not in cat:
                    continue
                elif category_filter.startswith("Danger") and "danger" not in cat:
                    continue
                elif category_filter.startswith("Mandatory") and "mandatory" not in cat:
                    continue
                elif category_filter.startswith("Other") and "other" not in cat:
                    continue
                matching_cids.append(cid)

            if not matching_cids:
                matching_cids = list(range(43))

            with c_filter2:
                selected_cid = st.selectbox(
                    "Select Traffic Sign Class (0 - 42)",
                    matching_cids,
                    format_func=lambda cid: f"[{cid:02d}] {GTSRB_CLASSES[cid]}"
                )

            with c_filter3:
                sample_var = st.radio("Sample Variant", [1, 2, 3], horizontal=True)

            candidate_name = f"class_{selected_cid:02d}_sample_{sample_var}.png"
            if os.path.exists(os.path.join(sample_dir, candidate_name)):
                chosen_file = candidate_name
            else:
                chosen_file = f"class_{selected_cid:02d}_sample_1.png"

        elif explorer_mode.startswith("Raw"):
            st.caption("Inspect uncurated test set slices directly from Test.csv (00000.png - 00029.png).")
            if general_samples:
                chosen_file = st.selectbox(
                    "Select Test.csv Image",
                    general_samples,
                    format_func=lambda f: f"{f} — Ground Truth: [{bench_loader.get_ground_truth(f).class_id if bench_loader.get_ground_truth(f) else '?'}] {bench_loader.get_ground_truth(f).class_name if bench_loader.get_ground_truth(f) else 'Unknown'}"
                )
            else:
                st.info("No raw test slices found in data/samples.")

        else:
            # Full Batch Benchmark Evaluation Suite
            st.markdown("##### ⚡ Automated 43-Class Benchmark Evaluation")
            st.caption("Execute an automated batch evaluation across all 129 canonical benchmark samples with active perception pipeline.")
            if st.button("🚀 Run Full 43-Class Evaluation", use_container_width=True):
                with st.spinner("Evaluating all 43 classes against active pipeline..."):
                    suite_files = sorted(glob.glob(os.path.join(sample_dir, "class_*.png")))
                    suite_total = len(suite_files)
                    suite_correct = 0
                    t_suite_start = time.perf_counter()
                    suite_results = []

                    for sf in suite_files:
                        s_gt = bench_loader.get_ground_truth(sf)
                        s_img = cv2.imread(sf)
                        s_res = pipeline.process_frame(s_img, conf_threshold=conf_thresh, is_video=False)
                        s_pred_id = s_res.detections[0].classification.class_id if s_res.detections else -1
                        s_pred_name = s_res.detections[0].classification.class_name if s_res.detections else "Unrecognized"
                        s_conf = s_res.detections[0].classification.confidence if s_res.detections else 0.0
                        is_match = (s_pred_id == s_gt.class_id) if s_gt else False
                        if is_match:
                            suite_correct += 1
                        suite_results.append({
                            "File": os.path.basename(sf),
                            "GT Class": f"[{s_gt.class_id}] {s_gt.class_name}" if s_gt else "N/A",
                            "Category": s_gt.category if s_gt else "N/A",
                            "Prediction": f"[{s_pred_id}] {s_pred_name}",
                            "Confidence": f"{s_conf*100:.1f}%",
                            "Status": "✅ Pass" if is_match else "❌ Fail"
                        })
                    suite_dur = (time.perf_counter() - t_suite_start) * 1000.0
                    acc = (suite_correct / max(1, suite_total)) * 100.0

                    m_acc, m_cnt, m_dur = st.columns(3)
                    m_acc.metric("Benchmark Accuracy", f"{acc:.1f}%")
                    m_cnt.metric("Passed / Total", f"{suite_correct} / {suite_total}")
                    m_dur.metric("Batch Execution Time", f"{suite_dur:.0f} ms ({suite_dur/suite_total:.1f} ms/sample)")

                    import pandas as pd
                    st.dataframe(pd.DataFrame(suite_results), use_container_width=True)
            chosen_file = None

        if chosen_file:
            img_path = os.path.join(sample_dir, chosen_file)
            frame = cv2.imread(img_path)
            gt = bench_loader.get_ground_truth(chosen_file)

            if frame is not None:
                result = pipeline.process_frame(frame, conf_threshold=conf_thresh, is_video=False)
                logger.log_detections(result.detections)
                render_audio_alerts(result, toggle_audio)
                if result.environmental_telemetry and result.environmental_telemetry.get("was_enhanced"):
                    st.caption(f"🌧️ Environmental Auto-Enhancement Active: {', '.join(result.environmental_telemetry.get('applied_ops', []))}")

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
            m_cols = st.columns(5 if "tt100k_ms" in result.latency_ms else 4)
            m_cols[0].metric("Signs Detected", result.num_signs_detected)
            m_cols[1].metric("Total Latency", f"{result.total_latency_ms:.1f} ms")
            m_cols[2].metric("Localization Time", f"{result.latency_ms['detect_ms']:.1f} ms")
            m_cols[3].metric("Classification Time", f"{result.latency_ms['classify_ms']:.1f} ms")
            if "tt100k_ms" in result.latency_ms:
                m_cols[4].metric("TT100K Secondary", f"{result.latency_ms['tt100k_ms']:.1f} ms")

            # TT100K Secondary Consensus Card
            if toggle_tt100k and result.detections:
                for d_idx, d in enumerate(result.detections):
                    if getattr(d, "tt100k_result", None) is not None:
                        tt = d.tt100k_result
                        status_color = "#15803d" if tt.is_consensus else "#b45309"
                        status_bg = "rgba(34, 197, 94, 0.08)" if tt.is_consensus else "rgba(245, 158, 11, 0.08)"
                        status_border = "rgba(34, 197, 94, 0.3)" if tt.is_consensus else "rgba(245, 158, 11, 0.3)"
                        with st.expander(f"🇨🇳 TT100K Secondary Consensus Telemetry (Sign #{d_idx+1}: {tt.class_code})", expanded=True):
                            st.markdown(f"""
                            <div style="background-color: {status_bg}; border: 1px solid {status_border}; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                                <div style="color: {status_color}; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">{tt.consensus_note}</div>
                                <div style="font-size: 1.1rem; font-weight: 600; margin-top: 4px;">Predicted TT100K Sign: <code>{tt.class_code}</code> — {tt.class_name}</div>
                                <div style="font-size: 0.88rem; color: #4B5563; margin-top: 2px;">TT100K Confidence: <b>{tt.confidence*100:.1f}%</b> • Equivalent GTSRB Class ID: <b>{tt.mapped_gtsrb_id if tt.mapped_gtsrb_id is not None else 'Unmapped'}</b></div>
                            </div>
                            """, unsafe_allow_html=True)
                            if tt.top_k:
                                import pandas as pd
                                st.dataframe(pd.DataFrame([{
                                    "Rank": i + 1,
                                    "TT100K Code": code,
                                    "Semantic Name": name,
                                    "Probability": f"{prob*100:.2f}%"
                                } for i, (code, name, prob) in enumerate(tt.top_k)]), use_container_width=True)

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
                render_audio_alerts(result, toggle_audio)
                if result.environmental_telemetry and result.environmental_telemetry.get("was_enhanced"):
                    st.caption(f"🌧️ Environmental Auto-Enhancement Active: {', '.join(result.environmental_telemetry.get('applied_ops', []))}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 📷 Original Image")
                    st.image(cv2.cvtColor(result.original_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                with c2:
                    st.markdown("##### 🎯 Detection & Classification HUD")
                    st.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                m_cols2 = st.columns(4 if "tt100k_ms" in result.latency_ms else 3)
                m_cols2[0].metric("Signs Detected", result.num_signs_detected)
                m_cols2[1].metric("Total Latency", f"{result.total_latency_ms:.1f} ms")
                m_cols2[2].metric("Throughput", f"{result.fps:.1f} FPS")
                if "tt100k_ms" in result.latency_ms:
                    m_cols2[3].metric("TT100K Latency", f"{result.latency_ms['tt100k_ms']:.1f} ms")

                # TT100K Secondary Consensus Card for Still Images
                if toggle_tt100k and result.detections:
                    for d_idx, d in enumerate(result.detections):
                        if getattr(d, "tt100k_result", None) is not None:
                            tt = d.tt100k_result
                            status_color = "#15803d" if tt.is_consensus else "#b45309"
                            status_bg = "rgba(34, 197, 94, 0.08)" if tt.is_consensus else "rgba(245, 158, 11, 0.08)"
                            status_border = "rgba(34, 197, 94, 0.3)" if tt.is_consensus else "rgba(245, 158, 11, 0.3)"
                            with st.expander(f"🇨🇳 TT100K Secondary Consensus Telemetry (Sign #{d_idx+1}: {tt.class_code})", expanded=True):
                                st.markdown(f"""
                                <div style="background-color: {status_bg}; border: 1px solid {status_border}; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                                    <div style="color: {status_color}; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">{tt.consensus_note}</div>
                                    <div style="font-size: 1.1rem; font-weight: 600; margin-top: 4px;">Predicted TT100K Sign: <code>{tt.class_code}</code> — {tt.class_name}</div>
                                    <div style="font-size: 0.88rem; color: #4B5563; margin-top: 2px;">TT100K Confidence: <b>{tt.confidence*100:.1f}%</b> • Equivalent GTSRB Class ID: <b>{tt.mapped_gtsrb_id if tt.mapped_gtsrb_id is not None else 'Unmapped'}</b></div>
                                </div>
                                """, unsafe_allow_html=True)
                                if tt.top_k:
                                    import pandas as pd
                                    st.dataframe(pd.DataFrame([{
                                        "Rank": i + 1,
                                        "TT100K Code": code,
                                        "Semantic Name": name,
                                        "Probability": f"{prob*100:.2f}%"
                                    } for i, (code, name, prob) in enumerate(tt.top_k)]), use_container_width=True)

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
    # OPTION 6: 100 COMMON ROAD SIGNS, COLOR TAXONOMY & OCR EXPLORER
    # =========================================================================
    elif app_mode.startswith("6."):
        st.subheader("🌐 Option 6: 100 Common Road Signs, Color Taxonomy & OCR Engine")
        st.markdown(
            "Explore the **100 Most Common International Road Signs** (Vienna Convention & MUTCD), "
            "the **10 Standard Color Combinations**, and test the **Plague Model Secondary Detector & OCR Engine**."
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 100 Common Road Signs Database",
            "🎨 10 Standard Color Combinations",
            "🦠 Plague Model & OCR Engine Playground",
            "🇨🇳 TT100K 221-Class Benchmark & Consensus"
        ])

        with tab1:
            from src.dataset.common_signs_100 import COMMON_ROAD_SIGNS_100
            import pandas as pd

            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                categories = ["All"] + sorted(list(set(s.category for s in COMMON_ROAD_SIGNS_100)))
                cat_filter = st.selectbox("Filter by Category", categories)
            with c_f2:
                has_num_filter = st.selectbox("Contains Numbers (Speed/Weight/Distance)", ["All", "Yes (Numbers Only)", "No"])
            with c_f3:
                has_txt_filter = st.selectbox("Contains Text / Word (STOP, ZONE, etc.)", ["All", "Yes (Text Only)", "No"])

            filtered_signs = COMMON_ROAD_SIGNS_100
            if cat_filter != "All":
                filtered_signs = [s for s in filtered_signs if s.category == cat_filter]
            if has_num_filter == "Yes (Numbers Only)":
                filtered_signs = [s for s in filtered_signs if s.has_number]
            elif has_num_filter == "No":
                filtered_signs = [s for s in filtered_signs if not s.has_number]
            if has_txt_filter == "Yes (Text Only)":
                filtered_signs = [s for s in filtered_signs if s.has_text]
            elif has_txt_filter == "No":
                filtered_signs = [s for s in filtered_signs if not s.has_text]

            df_signs = pd.DataFrame([{
                "ID": s.id,
                "Code": s.code,
                "Sign Name": s.name,
                "Category": s.category,
                "Shape": s.shape,
                "Colors (Pri / Sec / Sym)": f"{s.primary_color} / {s.secondary_color} / {s.symbol_color}",
                "Text / Number Content": f"Text: {s.text_content or '-'}" if s.has_text else (f"Number: {s.number_content or '-'}" if s.has_number else "-"),
                "Description": s.description
            } for s in filtered_signs])

            st.dataframe(df_signs, use_container_width=True, height=450)
            st.caption(f"Showing {len(filtered_signs)} of {len(COMMON_ROAD_SIGNS_100)} standardized international traffic signs.")

        with tab2:
            from src.detection.color_taxonomy import ROAD_SIGN_COLOR_COMBINATIONS

            st.markdown("#### The 10 Most Common Traffic Sign Color Combinations Worldwide")
            for rule in ROAD_SIGN_COLOR_COMBINATIONS:
                with st.expander(f"🎨 **{rule.primary_color} + {rule.secondary_color}** ({rule.combination_id}) — {rule.semantic_meaning}", expanded=False):
                    c_col1, c_col2 = st.columns([1, 2])
                    with c_col1:
                        st.markdown(f"**Primary Palette**: `{rule.primary_color}`")
                        st.markdown(f"**Secondary Palette**: `{rule.secondary_color}`")
                        st.markdown(f"**Accent / Symbol**: `{rule.accent_or_symbol_color}`")
                        st.markdown(f"**Associated Categories**: `{', '.join(rule.associated_categories)}`")
                    with c_col2:
                        st.markdown(f"**Semantic Rule**: {rule.semantic_meaning}")
                        st.markdown(f"**Representative Signs**: {', '.join(rule.representative_examples)}")

        with tab3:
            st.markdown("#### 🦠 Plague Model & OCR Engine Playground")
            st.markdown(
                "Upload any image crop or scene to test the **Plague Seed Spreading Model**, "
                "**Immune System Color Cancellation**, and the **Standalone Number / Alphabet OCR Engine**."
            )

            ocr_file = st.file_uploader("Upload Crop to Run Plague & OCR Analysis", type=["png", "jpg", "jpeg", "webp"], key="plague_test_uploader")
            if ocr_file:
                c_bytes = np.asarray(bytearray(ocr_file.read()), dtype=np.uint8)
                crop_img = cv2.imdecode(c_bytes, cv2.IMREAD_COLOR)
                if crop_img is not None:
                    c_p1, c_p2 = st.columns(2)
                    with c_p1:
                        st.markdown("##### 📷 Uploaded Image")
                        st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                    with c_p2:
                        st.markdown("##### 🔬 Secondary Diagnostic Analysis")
                        from src.detection.ocr_engine import RoadSignOCREngine
                        from src.detection.plague_detector import PlagueSecondaryDetector

                        plague_eng = PlagueSecondaryDetector(enable_ocr=True)
                        plague_results = plague_eng.detect(crop_img, conf_threshold=0.25)

                        ocr_eng = RoadSignOCREngine()
                        ocr_res = ocr_eng.detect(crop_img)

                        if ocr_res:
                            st.success(f"**OCR Detection Confirmed**: `{ocr_res.raw_string}` ({ocr_res.sign_type}, Confidence: {ocr_res.confidence*100:.1f}%)")
                            if ocr_res.detected_number:
                                st.metric("Recognized Numerical Value", f"{ocr_res.detected_number}")
                            if ocr_res.detected_word:
                                st.metric("Recognized Keyword / Word", f"{ocr_res.detected_word}")
                        else:
                            st.info("No direct OCR text/number glyph recognized inside central interior.")

                        if plague_results:
                            st.markdown(f"**Plague Spread Regions Found**: `{len(plague_results)}`")
                            for p_det, p_cls in plague_results:
                                st.markdown(f"- **[{p_cls.class_id}] {p_cls.class_name}** (Confidence: {p_cls.confidence*100:.1f}%) at `{p_det.bbox.to_xyxy()}`")
                        else:
                            st.warning("Plague model immune system canceled this region (no valid color seed, non-sign texture, or skin tone detected).")

        with tab4:
            st.markdown("#### 🇨🇳 Tsinghua-Tencent 100K (TT100K) Secondary Benchmark & Consensus Engine")
            st.markdown(
                "TT100K features **100,000 high-resolution street-view images** and **30,000 traffic sign instances** "
                "spanning **221 categories** (45 core benchmark classes). Operates as a secondary cross-domain validator."
            )

            from src.classification.tt100k_taxonomy import TT100K_CLASSES, TT100K_DESCRIPTIONS, TT100K_TO_GTSRB_MAP, get_tt100k_name, map_tt100k_to_gtsrb
            from src.classification.tt100k_model import get_tt100k_classifier
            import pandas as pd

            c_tt_sub1, c_tt_sub2 = st.tabs(["📊 221-Class Ontology & Cross-Domain Mapping", "🔬 TT100K Inference Playground"])

            with c_tt_sub1:
                cat_filter = st.selectbox(
                    "Filter TT100K Category",
                    ["All Categories (221 Classes)", "Prohibitory (pl*, pm*, pn, p*, pa*, pr*)", "Warning / Danger (w*)", "Indicatory & Mandatory (i*, il*, ip)"],
                    key="tt_cat_filter"
                )

                filtered_codes = []
                for code in TT100K_CLASSES:
                    if cat_filter.startswith("Prohibitory") and not (code.startswith("p") or code.startswith("pr") or code.startswith("pl") or code.startswith("pm")):
                        continue
                    elif cat_filter.startswith("Warning") and not code.startswith("w"):
                        continue
                    elif cat_filter.startswith("Indicatory") and not (code.startswith("i") or code.startswith("il") or code.startswith("m")):
                        continue
                    filtered_codes.append(code)

                df_tt = pd.DataFrame([{
                    "Index": idx,
                    "TT100K Code": code,
                    "Sign Name & Description": get_tt100k_name(code),
                    "Mapped GTSRB ID": map_tt100k_to_gtsrb(code) if map_tt100k_to_gtsrb(code) is not None else "Unmapped",
                    "Category": "Prohibitory" if code.startswith("p") else ("Warning" if code.startswith("w") else ("Indicatory" if code.startswith("i") else "Other"))
                } for idx, code in enumerate(filtered_codes)])

                st.dataframe(df_tt, use_container_width=True, height=400)
                st.caption(f"Displaying {len(filtered_codes)} of 221 total TT100K classes.")

            with c_tt_sub2:
                st.markdown("##### Test Crops Against the TT100K 221-Class Neural Network")
                tt_samples = sorted(glob.glob("data/tt100k/samples/*.jpg") + glob.glob("data/tt100k/samples/*.png"))

                col_sample, col_up = st.columns([1.2, 2])
                test_crop = None
                with col_sample:
                    st.markdown("**Sample Images (HuggingFace TT100K)**:")
                    if tt_samples:
                        chosen_tt_sample = st.selectbox("Select Sample", [os.path.basename(p) for p in tt_samples], key="tt_sample_sel")
                        if chosen_tt_sample:
                            s_path = os.path.join("data", "tt100k", "samples", chosen_tt_sample)
                            test_crop = cv2.imread(s_path)
                    else:
                        st.caption("No sample files found in data/tt100k/samples/")

                with col_up:
                    st.markdown("**Or Upload Custom Crop**:")
                    uploaded_tt = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="tt_up_test")
                    if uploaded_tt:
                        t_bytes = np.asarray(bytearray(uploaded_tt.read()), dtype=np.uint8)
                        test_crop = cv2.imdecode(t_bytes, cv2.IMREAD_COLOR)

                if test_crop is not None:
                    c_img, c_res = st.columns([1, 2])
                    with c_img:
                        st.image(cv2.cvtColor(test_crop, cv2.COLOR_BGR2RGB), caption=f"Input Crop ({test_crop.shape[1]}x{test_crop.shape[0]})", width=180)
                    with c_res:
                        tt_clf = get_tt100k_classifier()
                        sim_gtsrb = st.selectbox("Simulate Primary GTSRB Prediction (for Consensus Check)", [None] + list(range(43)), index=0, format_func=lambda x: f"GTSRB Class [{x}]" if x is not None else "None (Standalone)")
                        tt_res = tt_clf.analyze_crop(test_crop, primary_gtsrb_id=sim_gtsrb)

                        status_color = "#15803d" if tt_res.is_consensus else "#b45309"
                        status_bg = "rgba(34, 197, 94, 0.08)" if tt_res.is_consensus else "rgba(245, 158, 11, 0.08)"
                        status_border = "rgba(34, 197, 94, 0.3)" if tt_res.is_consensus else "rgba(245, 158, 11, 0.3)"

                        st.markdown(f"""
                        <div style="background-color: {status_bg}; border: 1px solid {status_border}; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                            <div style="color: {status_color}; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">{tt_res.consensus_note}</div>
                            <div style="font-size: 1.15rem; font-weight: 600; margin-top: 4px;">Top TT100K Class: <code>{tt_res.class_code}</code> — {tt_res.class_name}</div>
                            <div style="font-size: 0.88rem; color: #4B5563; margin-top: 2px;">Confidence: <b>{tt_res.confidence*100:.1f}%</b> • Mapped GTSRB ID: <b>{tt_res.mapped_gtsrb_id if tt_res.mapped_gtsrb_id is not None else 'Unmapped'}</b></div>
                        </div>
                        """, unsafe_allow_html=True)

                        if tt_res.top_k:
                            st.markdown("**Top-5 Cross-Domain Hypotheses**:")
                            st.dataframe(pd.DataFrame([{
                                "Rank": i + 1,
                                "TT100K Code": code,
                                "Semantic Name": name,
                                "Probability": f"{prob*100:.2f}%"
                            } for i, (code, name, prob) in enumerate(tt_res.top_k)]), use_container_width=True)

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
