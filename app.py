"""
Streamlit Web Dashboard for Road / Traffic Sign Detection & Recognition.
Features:
  1. Test Against Established Samples (GTSRB Benchmark)
  2. Test Against Still Images (File Upload)
  3. Continuous Real-Time Video Testing via Camera / Webcam
  4. Continuous Video Testing via Screenshare (Watches any video playing on your laptop)
  5. Test Against YouTube Video Links
  - Live Local Timestamped Event Stream (e.g. 03/10/2026 16:20:05 -> Stop sign detected)
  - Custom Dataset Uploader & On-the-Fly PyTorch Trainer
Maintained by Member D (Integration & Pipeline Lead).
"""

from typing import List, Dict, Optional
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
from src.schema import SignCategory, PipelineResult
from src.utils.screen_capture import ScreenCaptureHandler
from src.utils.youtube import YouTubeStreamHandler
from src.utils.event_logger import DetectionEventLogger
from src.dataset.custom_dataset import CustomDatasetManager

st.set_page_config(
    page_title="Traffic Sign Detection & Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .badge-prohibitory { background-color: #EF4444; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }
    .badge-danger { background-color: #F59E0B; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }
    .badge-mandatory { background-color: #3B82F6; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }
    .badge-other { background-color: #10B981; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; }
    .log-box {
        background-color: #0F172A;
        color: #38BDF8;
        font-family: monospace;
        font-size: 0.85rem;
        padding: 12px;
        border-radius: 6px;
        max-height: 220px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline(mode: str, padding: float):
    return TrafficSignPipeline.create(mode=mode, crop_padding_ratio=padding, enable_tracking=True)


# Initialize Session State for Event Logger
if "event_logger" not in st.session_state:
    st.session_state.event_logger = DetectionEventLogger(dedup_cooldown_sec=2.5)


def render_event_log_ui(event_logger: DetectionEventLogger):
    st.markdown("### ⏱️ Live Detection Event Log (User Local Time)")
    recent_logs = event_logger.get_recent_logs(limit=25)

    if recent_logs:
        log_html = "<div class='log-box'>" + "<br>".join(recent_logs) + "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

        col_csv, col_json, col_clr = st.columns([1, 1, 3])
        with col_csv:
            st.download_button(
                "📥 Download CSV Log",
                data=event_logger.export_csv(),
                file_name=f"sign_detections_{int(time.time())}.csv",
                mime="text/csv"
            )
        with col_json:
            st.download_button(
                "📥 Download JSON Log",
                data=event_logger.export_json(),
                file_name=f"sign_detections_{int(time.time())}.json",
                mime="application/json"
            )
        with col_clr:
            if st.button("🗑️ Clear Log"):
                event_logger.clear()
                st.rerun()
    else:
        st.info("No detection events logged yet. Detections will appear here in real-time timestamped with your local time.")


def main():
    st.markdown('<div class="main-header">🚦 Road / Traffic Sign Detection & Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unified ADAS Pipeline • Continuous-Time Video • GTSRB & Custom Datasets • Built by <b>Member D</b></div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Pipeline Configuration")

    mode_options = {
        "Production (Trained PyTorch CNN - 99.2% Acc)": "production",
        "Heuristic / CV (Fast Color-Contour Fallback)": "heuristic",
        "Mock / Dummy (Fast Verification)": "mock",
    }
    selected_mode_label = st.sidebar.selectbox("Model & Pipeline Mode", list(mode_options.keys()), index=0)
    mode = mode_options[selected_mode_label]

    conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.60, step=0.05)
    padding_ratio = st.sidebar.slider("Crop Margin Padding", min_value=0.0, max_value=0.20, value=0.05, step=0.02)

    # Teammate status indicators
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤝 System Status")
    yolo_exists = os.path.exists(os.path.join("weights", "detection", "best.pt"))
    cnn_exists = os.path.exists(os.path.join("weights", "classification", "classifier.pt"))

    st.sidebar.markdown(f"**Detector**: {'🟢 Trained YOLO' if yolo_exists else '🟡 Heuristic CV Active'}")
    st.sidebar.markdown(f"**Classifier**: {'🟢 PyTorch CNN (99.2%)' if cnn_exists else '🟡 Heuristic Active'}")
    st.sidebar.markdown("**Integration & Tracking**: 🟢 Active")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Testing Mode")
    app_mode = st.sidebar.radio(
        "Choose Option:",
        [
            "1. Test Against Established Samples",
            "2. Test Against Still Images",
            "3. Continuous Real-Time Video (Camera)",
            "4. Continuous Video (Screenshare / Laptop Video)",
            "5. Test Against YouTube Video Link",
            "📁 Custom Dataset & Training"
        ]
    )

    pipeline = get_pipeline(mode, padding_ratio)
    logger = st.session_state.event_logger

    # =========================================================================
    # OPTION 1: TEST AGAINST ESTABLISHED SAMPLES
    # =========================================================================
    if app_mode == "1. Test Against Established Samples":
        st.subheader("📊 Option 1: Test Against Established Samples (GTSRB Benchmark)")
        st.write("Browse and evaluate the model against pre-extracted test samples from the GTSRB dataset.")

        sample_dir = os.path.join("data", "samples")
        sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.png")))

        if not sample_files:
            st.warning("Sample dataset not found. Click below to extract sample images from archive.zip.")
            if st.button("Extract GTSRB Samples"):
                from scripts.prepare_sample_data import extract_samples
                extract_samples()
                st.rerun()
            return

        chosen_file = st.selectbox("Select Benchmark Test Sample", [os.path.basename(f) for f in sample_files])
        img_path = os.path.join(sample_dir, chosen_file)
        frame = cv2.imread(img_path)

        from src.dataset.benchmark_loader import BenchmarkDataLoader
        bench_loader = BenchmarkDataLoader()
        gt = bench_loader.get_ground_truth(chosen_file)

        if frame is not None:
            result = pipeline.process_frame(frame, conf_threshold=conf_thresh)
            logger.log_detections(result.detections)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Raw Test Image")
                st.image(cv2.cvtColor(result.original_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("🎯 Model Prediction & HUD")
                st.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Ground Truth vs Prediction Comparison Card
            if gt is not None:
                st.markdown("#### 📋 Benchmark Ground Truth Comparison (from `Test.csv` & `Meta.csv`)")
                gt_col1, gt_col2, gt_col3 = st.columns(3)
                gt_col1.markdown(f"**Ground Truth Label**:<br>`[{gt.class_id}] {gt.class_name}`", unsafe_allow_html=True)
                gt_col1.caption(f"Category: **{gt.category}** | Shape ID: {gt.shape_id} | Color ID: {gt.color_id}")
                gt_col2.markdown(f"**Ground Truth ROI Box**:<br>`({gt.roi_x1}, {gt.roi_y1}, {gt.roi_x2}, {gt.roi_y2})`", unsafe_allow_html=True)
                gt_col2.caption(f"Resolution: {gt.width}x{gt.height} px")

                pred_id = result.detections[0].classification.class_id if result.detections else -1
                pred_name = result.detections[0].classification.class_name if result.detections else "None"
                pred_conf = result.detections[0].classification.confidence if result.detections else 0.0

                gt_col3.markdown(f"**Model Prediction**:<br>`[{pred_id}] {pred_name}`", unsafe_allow_html=True)
                gt_col3.caption(f"Confidence: **{pred_conf*100:.1f}%**")

                if pred_id == gt.class_id:
                    st.success(f"✅ **EXACT BENCHMARK MATCH**: Model correctly recognized `{gt.class_name}` with {pred_conf*100:.1f}% confidence!")
                elif pred_id >= 0:
                    st.warning(f"⚠️ Predicted `{pred_name}` vs Ground Truth `{gt.class_name}`.")
                else:
                    st.info("ℹ️ Detection threshold filtered this crop. Adjust confidence threshold slider in sidebar.")

            # Telemetry Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Signs Detected", result.num_signs_detected)
            m2.metric("Inference Latency", f"{result.total_latency_ms} ms")
            m3.metric("Detect Time", f"{result.latency_ms['detect_ms']} ms")
            m4.metric("Classify Time", f"{result.latency_ms['classify_ms']} ms")

            render_event_log_ui(logger)

    # =========================================================================
    # OPTION 2: TEST AGAINST STILL IMAGES
    # =========================================================================
    elif app_mode == "2. Test Against Still Images":
        st.subheader("🖼️ Option 2: Test Against Still Images (File Upload)")
        st.write("Upload any still image from your computer to detect and recognize road signs.")

        uploaded = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame is not None:
                result = pipeline.process_frame(frame, conf_threshold=conf_thresh)
                logger.log_detections(result.detections)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📷 Uploaded Image")
                    st.image(cv2.cvtColor(result.original_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.subheader("🎯 Annotated Detections")
                    st.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Signs Detected", result.num_signs_detected)
                m2.metric("Total Latency", f"{result.total_latency_ms} ms")
                m3.metric("Throughput", f"{result.fps} FPS")

                render_event_log_ui(logger)

    # =========================================================================
    # OPTION 3: CONTINUOUS REAL-TIME VIDEO (CAMERA)
    # =========================================================================
    elif app_mode == "3. Continuous Real-Time Video (Camera)":
        st.subheader("📹 Option 3: Continuous Real-Time Video Testing (Camera / Webcam)")
        st.write("Connect to your webcam or vehicle dashcam for real-time continuous detection, temporal tracking, and ADAS HUD.")

        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            cam_idx = st.number_input("Camera Index (0 for primary webcam)", min_value=0, max_value=5, value=0)
        with col_c2:
            target_reticle = st.checkbox("🎯 Center Sign Focus Mode (Recommended when holding phone/paper sign to camera)", value=True)

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

                log_placeholder = st.empty()

                while run_cam and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    h, w = frame.shape[:2]

                    # Process raw frame through pipeline first
                    result = pipeline.process_frame(frame, conf_threshold=conf_thresh)

                    # If Center Reticle is enabled and no signs found by YOLO, evaluate center target box
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
                                # Highlight center reticle in green with label
                                reticle_color = (0, 255, 0)
                                cv2.rectangle(result.annotated_frame, (cx1, cy1), (cx2, cy2), reticle_color, 2, cv2.LINE_AA)
                                cv2.putText(
                                    result.annotated_frame,
                                    f"HOLD: {cls_res.class_name} ({cls_res.confidence*100:.0f}%)",
                                    (cx1 + 10, cy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, reticle_color, 2, cv2.LINE_AA
                                )
                                # Log detection if not already present
                                from src.schema import PipelineDetection
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
                    m_det.metric("Detect Time", f"{result.latency_ms['detect_ms']} ms")
                    m_cls.metric("Classify Time", f"{result.latency_ms['classify_ms']} ms")
                    m_fps.metric("Stream FPS", f"{result.fps} FPS")

                    time.sleep(0.01)

                cap.release()

        render_event_log_ui(logger)

    # =========================================================================
    # OPTION 4: CONTINUOUS VIDEO THROUGH SCREENSHARE (LAPTOP VIDEO)
    # =========================================================================
    elif app_mode == "4. Continuous Video (Screenshare / Laptop Video)":
        st.subheader("🖥️ Option 4: Continuous Video Through Screenshare (Laptop Screen)")
        st.markdown(
            "Plays alongside any video running on your laptop (YouTube in a browser, media player, dashcam recording). "
            "The system **watches your screen in real time** and detects traffic signs continuously!"
        )

        st.info("💡 **How to test**: Open any video on your screen (or open a browser with dashcam footage). Click 'Start Screen Capture Detection' below.")

        screen_choice = st.radio("Capture Area", ["Full Primary Monitor", "Custom Screen Region"], horizontal=True)
        region = None
        if screen_choice == "Custom Screen Region":
            col_x, col_y, col_w, col_h = st.columns(4)
            left = col_x.number_input("Left X", value=100, step=50)
            top = col_y.number_input("Top Y", value=100, step=50)
            width = col_w.number_input("Width", value=800, step=50)
            height = col_h.number_input("Height", value=500, step=50)
            region = (int(left), int(top), int(width), int(height))

        run_screen = st.toggle("▶️ Start Screen Capture Detection", value=False)

        if run_screen:
            handler = ScreenCaptureHandler(region=region)
            st_screen = st.empty()
            m_cols = st.columns(5)
            m_signs = m_cols[0].empty()
            m_speed = m_cols[1].empty()
            m_det = m_cols[2].empty()
            m_cls = m_cols[3].empty()
            m_fps = m_cols[4].empty()

            try:
                for frame in handler.frames(target_fps=20):
                    if not run_screen:
                        break

                    # Downsample slightly if screen is 4K to maintain high FPS
                    if frame.shape[1] > 1280:
                        scale = 1280.0 / frame.shape[1]
                        frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale)))

                    result = pipeline.process_frame(frame, conf_threshold=conf_thresh)
                    logger.log_detections(result.detections)

                    st_screen.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    m_signs.metric("Visible Signs", result.num_signs_detected)
                    m_speed.metric("Active Speed", result.active_speed_limit or "None")
                    m_det.metric("Detect Time", f"{result.latency_ms['detect_ms']} ms")
                    m_cls.metric("Classify Time", f"{result.latency_ms['classify_ms']} ms")
                    m_fps.metric("Stream FPS", f"{result.fps} FPS")

            finally:
                handler.close()

        render_event_log_ui(logger)

    # =========================================================================
    # OPTION 5: TEST AGAINST YOUTUBE VIDEO LINKS
    # =========================================================================
    elif app_mode == "5. Test Against YouTube Video Link":
        st.subheader("🌐 Option 5: Test Against YouTube Video Links")
        st.write("Paste a link to any driving / dashcam video on YouTube. The system streams the video and recognizes traffic signs in continuous time.")

        default_yt = "https://www.youtube.com/watch?v=0kG2gU-uA6Q"
        yt_url = st.text_input("Enter YouTube Video URL", value="")

        st.caption("Examples: Driving in Germany Autobahn, dashcam videos, or traffic test clips.")

        if yt_url:
            yt_handler = YouTubeStreamHandler()

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                fetch_stream = st.button("🔍 Resolve & Stream Directly")
            with col_btn2:
                download_clip = st.button("⬇️ Download 30s Clip for Local Fast Playback")

            video_source = None

            if download_clip:
                with st.spinner("Downloading clip via yt-dlp..."):
                    local_clip = yt_handler.download_video_segment(yt_url, output_filename="yt_dashcam.mp4")
                    if local_clip and os.path.exists(local_clip):
                        st.success(f"Clip downloaded successfully to {local_clip}!")
                        st.session_state["yt_source"] = local_clip
                    else:
                        st.error("Failed to download YouTube clip. Direct streaming will be used instead.")

            if fetch_stream:
                with st.spinner("Extracting stream URL via yt-dlp..."):
                    stream_url = yt_handler.get_direct_stream_url(yt_url)
                    if stream_url:
                        st.session_state["yt_source"] = stream_url
                    else:
                        st.error("Could not extract stream URL. Try using Option 4 (Screenshare) while playing the video!")

            active_source = st.session_state.get("yt_source")
            if active_source:
                run_yt = st.toggle("▶️ Start YouTube Detection Stream", value=False)
                if run_yt:
                    cap = cv2.VideoCapture(active_source)
                    st_yt = st.empty()
                    m_cols = st.columns(5)
                    m_signs = m_cols[0].empty()
                    m_speed = m_cols[1].empty()
                    m_det = m_cols[2].empty()
                    m_cls = m_cols[3].empty()
                    m_fps = m_cols[4].empty()

                    while run_yt and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.info("Reached end of YouTube stream.")
                            break

                        result = pipeline.process_frame(frame, conf_threshold=conf_thresh)
                        logger.log_detections(result.detections)

                        st_yt.image(cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        m_signs.metric("Visible Signs", result.num_signs_detected)
                        m_speed.metric("Active Speed", result.active_speed_limit or "None")
                        m_det.metric("Detect Time", f"{result.latency_ms['detect_ms']} ms")
                        m_cls.metric("Classify Time", f"{result.latency_ms['classify_ms']} ms")
                        m_fps.metric("Stream FPS", f"{result.fps} FPS")

                        time.sleep(0.01)

                    cap.release()

        render_event_log_ui(logger)

    # =========================================================================
    # OPTION 6: CUSTOM DATASET & TRAINING
    # =========================================================================
    elif app_mode == "📁 Custom Dataset & Training":
        st.subheader("📁 Custom Dataset Management & On-The-Fly Model Training")
        st.markdown(
            "Easily train a custom PyTorch model for **regional traffic signs** (e.g., Indian signs, US MUTCD signs, or industrial symbols) "
            "without touching code!"
        )

        dataset_mgr = CustomDatasetManager()

        st.markdown("#### 1. Upload Custom Dataset (ZIP)")
        st.markdown("Folder structure inside ZIP should be folder-per-class (e.g. `speed_40/`, `stop_sign/`, `yield/`).")
        uploaded_zip = st.file_uploader("Upload Dataset (.zip)", type=["zip"])

        if uploaded_zip:
            temp_zip = os.path.join("data", "custom", "uploaded_dataset.zip")
            with open(temp_zip, "wb") as f:
                f.write(uploaded_zip.read())

            with st.spinner("Extracting and indexing custom classes..."):
                unpacked_dir = dataset_mgr.unpack_zip(temp_zip)
                summary = dataset_mgr.get_summary(unpacked_dir)
                st.success(f"Dataset extracted! Found {summary['num_classes']} classes and {summary['total_images']} total images.")

                with st.expander("View Detected Classes & Distribution"):
                    st.json(summary["class_distribution"])

                st.markdown("#### 2. Train Custom PyTorch CNN Model")
                epochs = st.slider("Training Epochs", min_value=2, max_value=20, value=6)
                lr = st.select_slider("Learning Rate", options=[0.0005, 0.001, 0.002, 0.005], value=0.001)

                if st.button("🚀 Train Model On Custom Dataset"):
                    prog_bar = st.progress(0.0)
                    status_text = st.empty()

                    # Load images from custom folders
                    custom_images = []
                    custom_labels = []
                    class_map = dataset_mgr.get_class_map()

                    for cid, cname in class_map.items():
                        c_folder = os.path.join(unpacked_dir, cname.replace(" ", "_").lower())
                        for p in glob.glob(os.path.join(c_folder, "*.*")):
                            img = cv2.imread(p)
                            if img is not None:
                                res = cv2.resize(img, (32, 32))
                                rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                                custom_images.append(np.transpose(rgb, (2, 0, 1)))
                                custom_labels.append(cid)

                    if len(custom_images) < 10:
                        st.error("Not enough images found in custom folders (minimum 10 images required).")
                    else:
                        X_cust = np.array(custom_images, dtype=np.float32)
                        y_cust = np.array(custom_labels, dtype=np.int64)

                        from src.train_custom import train_model

                        def callback(current_epoch, total_epochs, train_acc, val_acc):
                            prog_bar.progress(current_epoch / float(total_epochs))
                            status_text.text(f"Epoch {current_epoch}/{total_epochs} - Accuracy: {train_acc*100:.1f}%")

                        acc, out_path = train_model(
                            train_images=X_cust,
                            train_labels=y_cust,
                            num_classes=len(class_map),
                            epochs=epochs,
                            learning_rate=lr,
                            output_path="weights/classification/custom_classifier.pt",
                            progress_callback=callback
                        )

                        st.success(f"🎉 Custom model trained with {acc*100:.1f}% accuracy! Saved to {out_path}.")
                        st.info("To use this model in the pipeline, select 'Custom Model' mode in the sidebar.")


if __name__ == "__main__":
    main()
