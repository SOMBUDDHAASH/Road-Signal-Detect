"""
FastAPI REST microservice for Traffic Sign Detection & Recognition.
Maintained by Member D (Integration & Pipeline Lead).
"""

from typing import List, Dict, Any
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from src.pipeline import TrafficSignPipeline

app = FastAPI(
    title="Road / Traffic Sign Detection API",
    description="REST backend for GTSRB Traffic Sign Detection and Recognition. Maintained by Member D.",
    version="1.0.0"
)

# Pipeline instances cached
pipelines: Dict[str, TrafficSignPipeline] = {}


def get_pipeline_instance(mode: str = "heuristic") -> TrafficSignPipeline:
    if mode not in pipelines:
        pipelines[mode] = TrafficSignPipeline.create(mode=mode)
    return pipelines[mode]


class DetectionResponse(BaseModel):
    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str
    category: str


class PredictResponse(BaseModel):
    signs_detected: int
    latency_ms: Dict[str, float]
    detections: List[DetectionResponse]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "traffic-sign-pipeline",
        "available_modes": ["mock", "heuristic", "production"]
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_image(
    file: UploadFile = File(...),
    mode: str = Query("heuristic", pattern="^(mock|heuristic|production)$"),
    conf_threshold: float = Query(0.50, ge=0.0, le=1.0)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pipeline = get_pipeline_instance(mode)
    result = pipeline.process_frame(image, conf_threshold=conf_threshold)

    detections_out = []
    for item in result.detections:
        detections_out.append(DetectionResponse(
            bbox=list(item.detection.bbox.to_xyxy()),
            confidence=round(item.classification.confidence, 4),
            class_id=item.classification.class_id,
            class_name=item.classification.class_name,
            category=item.classification.category.value
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
    conf_threshold: float = Query(0.50, ge=0.0, le=1.0)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pipeline = get_pipeline_instance(mode)
    result = pipeline.process_frame(image, conf_threshold=conf_threshold)

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
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
