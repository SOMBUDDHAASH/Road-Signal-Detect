"""
Unit tests for the FastAPI REST interface.
"""

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "heuristic" in data["available_modes"]


def test_api_predict_mock():
    # Send a dummy black 100x100 PNG
    import cv2
    import numpy as np
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", dummy_img)

    response = client.post(
        "/predict",
        files={"file": ("dummy.png", encoded.tobytes(), "image/png")},
        params={"mode": "mock"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "signs_detected" in data
    assert "latency_ms" in data
    assert len(data["detections"]) >= 1


def test_api_predict_annotated():
    import cv2
    import numpy as np
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", dummy_img)

    response = client.post(
        "/predict/annotated",
        files={"file": ("dummy.png", encoded.tobytes(), "image/png")},
        params={"mode": "mock"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0
