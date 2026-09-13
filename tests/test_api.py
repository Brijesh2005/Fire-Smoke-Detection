"""
Unit and Integration Tests for Fire & Smoke Detection Web Application.
"""

import os
import io
import base64
import pytest
from app import app
from model_service import FireSmokeDetector

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_detector_singleton():
    detector1 = FireSmokeDetector.get_instance()
    detector2 = FireSmokeDetector.get_instance()
    assert detector1 is detector2
    assert detector1.CLASS_NAMES == ["Fire", "Neutral", "Smoke"]

def test_predict_known_samples():
    detector = FireSmokeDetector.get_instance()

    # Smoke sample
    smoke_path = os.path.join("test-imgs", "image_0.jpg")
    if os.path.exists(smoke_path):
        with open(smoke_path, "rb") as f:
            res = detector.predict_bytes(f.read())
        assert res["prediction"] == "Smoke"
        assert res["is_hazard"] is True
        assert res["confidence"] > 90.0

    # Fire sample
    fire_path = os.path.join("test-imgs", "26.jpg")
    if os.path.exists(fire_path):
        with open(fire_path, "rb") as f:
            res = detector.predict_bytes(f.read())
        assert res["prediction"] == "Fire"
        assert res["is_hazard"] is True
        assert res["confidence"] > 90.0

    # Neutral sample
    neutral_path = os.path.join("test-imgs", "image_7.jpg")
    if os.path.exists(neutral_path):
        with open(neutral_path, "rb") as f:
            res = detector.predict_bytes(f.read())
        assert res["prediction"] == "Neutral"
        assert res["is_hazard"] is False

def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["model"] == "ResNet-50 (Transfer Learning)"
    assert "classes" in data
    assert "Fire" in data["classes"]

def test_api_samples_list(client):
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples = res.get_json()
    assert isinstance(samples, list)
    assert len(samples) > 0
    assert any(s["filename"] == "image_0.jpg" for s in samples)

def test_api_predict_sample_json(client):
    res = client.post("/api/predict/image", json={"sample_filename": "image_0.jpg"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["prediction"] == "Smoke"
    assert data["is_hazard"] is True
    assert "probabilities" in data

def test_api_predict_upload_file(client):
    smoke_path = os.path.join("test-imgs", "image_0.jpg")
    with open(smoke_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "file": (io.BytesIO(img_bytes), "image_0.jpg")
    }
    res = client.post("/api/predict/image", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    result = res.get_json()
    assert result["prediction"] == "Smoke"
    assert result["confidence"] > 90.0

def test_api_predict_frame_base64(client):
    smoke_path = os.path.join("test-imgs", "image_0.jpg")
    with open(smoke_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    res = client.post("/api/predict/frame", json={"frame": f"data:image/jpeg;base64,{b64}"})
    assert res.status_code == 200
    result = res.get_json()
    assert result["prediction"] == "Smoke"

def test_api_predict_video(client):
    import cv2
    temp_video = "tests_temp_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_video, fourcc, 10, (320, 240))
    img = cv2.imread("test-imgs/image_0.jpg")
    img = cv2.resize(img, (320, 240))
    for _ in range(10):
        writer.write(img)
    writer.release()

    try:
        with open(temp_video, "rb") as f:
            data = {"file": (f, "test_clip.mp4"), "sample_interval": "0.5"}
            res = client.post("/api/predict/video", data=data, content_type="multipart/form-data")
        assert res.status_code == 200
        result = res.get_json()
        assert result["status"] == "success"
        assert result["overall_prediction"] == "Smoke"
        assert "timeline" in result
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)

