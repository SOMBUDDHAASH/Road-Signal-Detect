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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
