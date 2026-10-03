# Utilities & Video Streaming Subsystems (`src/utils/`)

This directory houses the visualization, screen capture, video processing, YouTube streaming, and event logging subsystems developed by **Member D (Integration Lead)**.

---

## 📄 File Index & Detailed Descriptions

### 1. `visualizer.py`
* **Purpose**: High-tech automotive Heads-Up Display (HUD) and bounding box annotation engine.
* **Key Visual Features**:
  * **Corner Accents & Target Brackets**: Renders sleek corner targeting brackets on detected signs instead of plain thick rectangles.
  * **Category-Color Coded Labels**: Displays semi-transparent pill headers matching sign category colors (Red for Prohibitory, Orange for Danger, Blue for Mandatory).
  * **Cockpit HUD Overlay**: Upper telemetry dashboard displaying System State, Pipeline Latency (`Det: XX ms`, `Cls: XX ms`), FPS, and Visible Sign counts.
  * **ADAS Hazard Banner**: Flashing high-visibility warning banner across the upper windshield when a danger sign is actively tracked.
  * **Active Speed Limit Dial**: Upper-left speed circle displaying the current legally enforced road speed.
* **Interconnections**: Invoked in Stage 4 of `TrafficSignPipeline.process_frame()` to render `PipelineResult.annotated_frame`.

---

### 2. `screen_capture.py`
* **Purpose**: Enables continuous real-time traffic sign detection directly from your laptop screen (Option 4).
* **Key Implementation**:
  * Uses the ultra-low-latency `mss` library to grab full-resolution monitor frames at **30+ FPS** without CPU bottlenecks.
  * Supports full monitor capture or custom bounding box extraction (e.g. centering on a video player or YouTube window).
* **Interconnections**: Used by Option 4 in `app.py`.

---

### 3. `youtube.py`
* **Purpose**: Fetches and streams public YouTube dashcam and driving videos into the detection pipeline (Option 5).
* **Key Implementation**:
  * Uses `yt-dlp` to extract direct progressive video streams without requiring third-party video downloading services.
  * Includes fallback to download short 15-second MP4 video clips locally for smooth 60 FPS offline playback.
* **Interconnections**: Used by Option 5 in `app.py`.

---

### 4. `event_logger.py`
* **Purpose**: Maintains a time-synchronized audit log of all traffic sign detections with the user's local timezone timecode.
* **Key Implementation**:
  * Formats events as: `DD/MM/YYYY HH:MM:SS -> [Class ID] Sign Name (Confidence%)`.
  * **De-duplication Cooldown**: Prevents spamming the event log when the car is stopped at a red light or stop sign (default 2.5s cooldown per sign class).
  * **Export**: Generates instant download payloads for **CSV** and **JSON** telemetry trip reports.
* **Interconnections**: Bound to Streamlit session state in `app.py` and displayed across all testing modes.

---

### 5. `video.py`
* **Purpose**: Standardized OpenCV video file and webcam capture wrapper with FPS calculation and frame resizing.
* **Interconnections**: Used in `src/live_feed.py` and video file playback.
