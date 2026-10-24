"""
FastAPI REST Microservice for Road Traffic Sign Detection & Recognition.
Unified service integrating:
  - Member A (Data Preprocessing & Augmentation)
  - Member B (YOLOv8 Detection)
  - Member C (Keras 3 / PyTorch CNN Classification)
  - Member D (Master Perception Pipeline, ADAS HUD, Telemetry, and TT100K Consensus)
"""

from typing import List, Dict, Any, Optional
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import TrafficSignPipeline
from src.config import CLASS_NAMES, SPEED_LIMIT_MAP

app = FastAPI(
    title="Road Traffic Sign Detection & Recognition API",
    description="Unified REST microservice for GTSRB 43-Class Perception with TT100K Secondary Analysis.",
    version="2.0.0"
)

# Enable CORS for cross-origin web client and dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pipeline instances cached
pipelines: Dict[str, TrafficSignPipeline] = {}


def get_pipeline_instance(mode: str = "heuristic") -> TrafficSignPipeline:
    if mode not in pipelines:
        pipelines[mode] = TrafficSignPipeline.create(mode=mode)
    return pipelines[mode]


class TT100KResponse(BaseModel):
    class_code: str
    class_name: str
    confidence: float
    mapped_gtsrb_id: Optional[int] = None
    is_consensus: bool = False
    consensus_note: str = ""


class DetectionResponse(BaseModel):
    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str
    category: str
    tt100k: Optional[TT100KResponse] = None


class PredictResponse(BaseModel):
    signs_detected: int
    latency_ms: Dict[str, float]
    detections: List[DetectionResponse]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "traffic-sign-pipeline",
        "version": "2.0.0",
        "available_modes": ["mock", "heuristic", "production"],
        "secondary_models": ["plague_detector", "tt100k_classifier"]
    }


@app.get("/classes")
def list_classes():
    """Returns official 43 GTSRB classes and speed limit mappings."""
    return {
        "total_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "speed_limits": SPEED_LIMIT_MAP
    }



@app.get("/tt100k/classes")
def list_tt100k_classes():
    from src.classification.tt100k_taxonomy import TT100K_CLASSES, TT100K_DESCRIPTIONS, TT100K_TO_GTSRB_MAP
    return {
        "total_classes": len(TT100K_CLASSES),
        "classes": TT100K_CLASSES,
        "descriptions": TT100K_DESCRIPTIONS,
        "cross_domain_gtsrb_mapping": TT100K_TO_GTSRB_MAP
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_image(
    file: UploadFile = File(...),
    mode: str = Query("heuristic", pattern="^(mock|heuristic|production)$"),
    conf_threshold: float = Query(0.50, ge=0.0, le=1.0),
    enable_tt100k: bool = Query(False)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pipeline = get_pipeline_instance(mode)
    result = pipeline.process_frame(image, conf_threshold=conf_threshold, is_video=False, enable_tt100k=enable_tt100k)

    detections_out = []
    for item in result.detections:
        tt_resp = None
        if getattr(item, "tt100k_result", None) is not None:
            tt = item.tt100k_result
            tt_resp = TT100KResponse(
                class_code=tt.class_code,
                class_name=tt.class_name,
                confidence=round(float(tt.confidence), 4),
                mapped_gtsrb_id=tt.mapped_gtsrb_id,
                is_consensus=tt.is_consensus,
                consensus_note=tt.consensus_note
            )
        detections_out.append(DetectionResponse(
            bbox=list(item.detection.bbox.to_xyxy()),
            confidence=round(item.classification.confidence, 4),
            class_id=item.classification.class_id,
            class_name=item.classification.class_name,
            category=item.classification.category.value,
            tt100k=tt_resp
        ))

    return PredictResponse(
        signs_detected=result.num_signs_detected,
        latency_ms=result.latency_ms,
        detections=detections_out
    )


@app.post("/predict/annotated")
async def predict_annotated_image(
    file: UploadFile = File(...),
    mode: str = Query("heuristic", pattern="^(mock|heuristic|production)$"),
    conf_threshold: float = Query(0.50, ge=0.0, le=1.0),
    enable_tt100k: bool = Query(False)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pipeline = get_pipeline_instance(mode)
    result = pipeline.process_frame(image, conf_threshold=conf_threshold, is_video=False, enable_tt100k=enable_tt100k)

    # Encode annotated frame as JPEG
    success, buffer = cv2.imencode(".jpg", result.annotated_frame)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/video_feed")
def live_mjpeg_stream(mode: str = Query("heuristic"), device_index: int = Query(0)):
    """
    Live MJPEG streaming endpoint for external displays, VLC, and vehicle dash displays.
    """
    from fastapi.responses import StreamingResponse

    def frame_generator():
        cap = cv2.VideoCapture(device_index)
        pipeline = get_pipeline_instance(mode)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result = pipeline.process_frame(frame, is_video=True)
                ret_jpg, jpeg = cv2.imencode(".jpg", result.annotated_frame)
                if ret_jpg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        finally:
            cap.release()

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


from fastapi import WebSocket, WebSocketDisconnect
import time
import base64

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    High-speed bi-directional telemetry WebSocket for vehicle fleet telemetry.
    Accepts base64 frame packets and streams instantaneous detections, active speed limits,
    entropy, and hazards back to the client.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if "image_base64" in data:
                img_bytes = base64.b64decode(data["image_base64"])
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    pipeline = get_pipeline_instance(data.get("mode", "heuristic"))
                    result = pipeline.process_frame(img, is_video=data.get("is_video", False))
                    await websocket.send_json({
                        "signs_detected": result.num_signs_detected,
                        "active_speed_limit": result.active_speed_limit,
                        "active_hazard": result.active_hazard,
                        "fps": result.fps,
                        "latency_ms": result.latency_ms,
                        "detections": [
                            {
                                "bbox": list(d.detection.bbox.to_xyxy()),
                                "class_id": d.classification.class_id,
                                "class_name": d.classification.class_name,
                                "confidence": round(d.classification.confidence, 3),
                                "category": d.classification.category.value,
                                "entropy": getattr(d.classification, "entropy", 0.0),
                                "margin": getattr(d.classification, "margin", 1.0),
                                "is_ambiguous": getattr(d.classification, "is_ambiguous", False)
                            } for d in result.detections
                        ]
                    })
                else:
                    await websocket.send_json({"error": "Failed to decode image"})
            else:
                await websocket.send_json({"status": "ready", "server_time": time.time()})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import os
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "false").lower() in ("true", "1")
    uvicorn.run("api:app", host=host, port=port, reload=reload)
