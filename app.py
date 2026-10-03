"""
Streamlit Web Dashboard for Road / Traffic Sign Detection & Recognition.
Maintained by Member D (Integration & Pipeline Lead).
"""

import os
import glob
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import TrafficSignPipeline
from src.gtsrb_classes import GTSRB_CLASSES, GTSRB_CATEGORIES, get_sign_category
from src.schema import SignCategory

st.set_page_config(
    page_title="Traffic Sign Detection & Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
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
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .badge-prohibitory {
        background-color: #EF4444; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;
    }
    .badge-danger {
        background-color: #F59E0B; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;
    }
    .badge-mandatory {
        background-color: #3B82F6; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;
    }
    .badge-other {
        background-color: #10B981; color: white; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline(mode: str, padding: float):
    return TrafficSignPipeline.create(mode=mode, crop_padding_ratio=padding)


def main():
    st.markdown('<div class="main-header">🚦 Road / Traffic Sign Detection & Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unified Integration Pipeline • GTSRB Benchmark (43 Classes) • Built by <b>Member D</b></div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Pipeline Configuration")

    mode_display = {
        "Heuristic / CV (Immediate Live Demo)": "heuristic",
        "Mock / Dummy (Fast Verification)": "mock",
        "Production (YOLO + PyTorch CNN)": "production",
    }
    selected_mode_label = st.sidebar.selectbox("Pipeline Mode", list(mode_display.keys()), index=0)
    mode = mode_display[selected_mode_label]

    conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
    padding_ratio = st.sidebar.slider("Crop Margin Padding", min_value=0.0, max_value=0.20, value=0.05, step=0.02)

    # Teammate status indicators
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤝 Team Integration Status")
    yolo_exists = os.path.exists(os.path.join("weights", "detection", "best.pt"))
    cnn_exists = os.path.exists(os.path.join("weights", "classification", "classifier.pt"))

    st.sidebar.markdown(f"**Member B (YOLO Detector)**: {'🟢 Ready (`best.pt`)' if yolo_exists else '🟡 Fallback (Heuristic Active)'}")
    st.sidebar.markdown(f"**Member C (CNN Classifier)**: {'🟢 Ready (`classifier.pt`)' if cnn_exists else '🟡 Fallback (Heuristic Active)'}")
    st.sidebar.markdown("**Member D (Pipeline)**: 🟢 Fully Operational")

    st.sidebar.markdown("---")
    input_source = st.sidebar.radio(
        "Input Source",
        ["Sample GTSRB Images", "Upload Image", "Camera Input"]
    )

    # Initialize Pipeline
    pipeline = get_pipeline(mode, padding_ratio)

    frame_to_process = None

    if input_source == "Sample GTSRB Images":
        sample_dir = os.path.join("data", "samples")
        sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
        if sample_files:
            file_names = [os.path.basename(f) for f in sample_files]
            chosen_file = st.selectbox("Select Sample Image", file_names, index=0)
            img_path = os.path.join(sample_dir, chosen_file)
            img_bgr = cv2.imread(img_path)
            frame_to_process = img_bgr
        else:
            st.warning("No sample images found in data/samples. Extracting...")

    elif input_source == "Upload Image":
        uploaded = st.file_uploader("Choose a traffic sign image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif input_source == "Camera Input":
        cam_pic = st.camera_input("Take a photo of a road sign")
        if cam_pic:
            file_bytes = np.asarray(bytearray(cam_pic.read()), dtype=np.uint8)
            frame_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame_to_process is not None:
        # Run Pipeline
        result = pipeline.process_frame(frame_to_process, conf_threshold=conf_thresh)

        # Telemetry metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Signs Detected", result.num_signs_detected)
        m2.metric("Detect Latency", f"{result.latency_ms['detect_ms']} ms")
        m3.metric("Classify Latency", f"{result.latency_ms['classify_ms']} ms")
        m4.metric("Total Latency", f"{result.total_latency_ms} ms")
        m5.metric("Throughput", f"{result.fps} FPS")

        st.markdown("---")

        # Two columns for Original vs Annotated
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("📷 Raw Input Frame")
            rgb_orig = cv2.cvtColor(result.original_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_orig, use_container_width=True)

        with col_right:
            st.subheader("🎯 Annotated Detections & HUD")
            rgb_annotated = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_annotated, use_container_width=True)

        # Sign Breakdown Gallery
        st.subheader(f"📋 Detected Signs Breakdown ({result.num_signs_detected})")
        if result.detections:
            cols = st.columns(min(4, max(1, len(result.detections))))
            for idx, item in enumerate(result.detections):
                col = cols[idx % len(cols)]
                with col:
                    if item.crop is not None and item.crop.size > 0:
                        crop_rgb = cv2.cvtColor(item.crop, cv2.COLOR_BGR2RGB)
                        st.image(crop_rgb, width=130)

                    cls = item.classification
                    badge_class = f"badge-{cls.category.value.lower()}"
                    st.markdown(f'<span class="{badge_class}">{cls.category.value}</span>', unsafe_allow_html=True)
                    st.markdown(f"**[{cls.class_id}] {cls.class_name}**")
                    st.progress(float(cls.confidence))
                    st.caption(f"Confidence: **{cls.confidence*100:.1f}%**")
                    
                    if cls.top_k:
                        with st.expander("Top Alternatives"):
                            for cid, name, score in cls.top_k[1:]:
                                st.write(f"- **{name}**: {score*100:.1f}%")
        else:
            st.info("No traffic signs detected at the selected confidence threshold. Try adjusting the slider.")

    # GTSRB Reference Tab
    with st.expander("📖 GTSRB Benchmark Reference (43 Classes)"):
        st.write("Full catalog of German Traffic Sign Recognition Benchmark classes:")
        cat_filter = st.selectbox("Filter Category", ["All", "Prohibitory", "Danger", "Mandatory", "Other"])
        
        filtered_classes = []
        for cid, name in GTSRB_CLASSES.items():
            cat = GTSRB_CATEGORIES[cid]
            if cat_filter == "All" or cat.value == cat_filter:
                filtered_classes.append((cid, name, cat.value))

        ref_cols = st.columns(3)
        for i, (cid, name, cat) in enumerate(filtered_classes):
            col = ref_cols[i % 3]
            col.write(f"`ID {cid:02d}`: **{name}** *({cat})*")


if __name__ == "__main__":
    main()
