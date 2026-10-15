# Automotive HUD & Streaming Subsystems (`src/utils/`)

This directory houses the visualization engine, screen capture driver, video streaming adapters, and real-time telemetry logging subsystems.

---

## 📄 File Index & Specifications

### 1. `visualizer.py`
* **Purpose**: Automotive Heads-Up Display (HUD) and bounding box annotation engine.
* **Key Features**:
  * **Target Brackets**: Sleek corner targeting brackets around detected signs.
  * **Category Badges**: Color-coded semi-transparent pill headers (Red for Prohibitory, Orange for Danger, Blue for Mandatory, Green for Other).
  * **Cockpit HUD Overlay**: Upper telemetry dashboard displaying System State, Pipeline Latencies (`Det: XX ms`, `Cls: XX ms`), FPS, and Visible Sign Counts.
  * **ADAS Hazard Banner**: Flashing alert banner when approaching actively tracked hazard signs.
  * **Active Speed Limit Dial**: Digital speedometer badge showing the current road speed restriction.

---

### 2. `screen_capture.py`
* **Purpose**: Ultra-low-latency desktop screen capture utilizing `mss` for real-time inference on media players, driving simulators, or video streams at **30+ FPS**.

---

### 3. `youtube.py`
* **Purpose**: Video ingestion driver for public dashcam recordings using `yt-dlp` progressive streaming and short-clip caching.

---

### 4. `event_logger.py`
* **Purpose**: Time-synchronized audit log recording sign perception events with local timestamps:
  * Formatted output: `DD/MM/YYYY HH:MM:SS -> [Class ID] Sign Name (Confidence%)`.
  * De-duplication cooldown to prevent log spamming during stationary vehicle stops.
  * Instant export to **CSV** and **JSON** telemetry trip reports.

---

### 5. `environmental.py`
* **Purpose**: Adaptive CLAHE (Contrast Limited Adaptive Histogram Equalization) and dark-channel prior atmospheric dehazing for adverse weather (night, rain, fog, glare).

---

### 6. `audio_alert.py`
* **Purpose**: Acoustic chime synthesizer and browser Web Speech API voice alert transducer (*"Stop sign ahead"*, *"Speed limit 50 km/h"*).
