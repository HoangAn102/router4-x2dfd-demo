import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from web.backend.app.main import app

client = TestClient(app)


def create_test_image(format="JPEG", size=(256, 256), color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "run_mode" in data
    assert data["run_mode"] == "mock"


def test_ready_endpoint():
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["all_ready"] is True
    assert "mock_engine" in data["components"]


def test_analyze_image_valid():
    img_bytes = create_test_image()
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["media_type"] == "image"
    assert data["router"]["selected_expert"] in ["blending", "diffusion", "frequency", "texture"]
    assert 0.0 <= data["final"]["continuous_score"] <= 1.0
    assert data["final"]["verdict"] in ["REAL", "FAKE"]
    assert data["provenance"]["mode"] == "mock_development"
    # Explanation is either str or None (never fabricated string)
    assert data["final"]["explanation"] is None or isinstance(data["final"]["explanation"], str)


def test_analyze_image_empty_file():
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "EMPTY_FILE"


def test_analyze_image_invalid_magic_bytes():
    # A text file renamed to .jpg
    fake_img = b"This is plain text, not an image!"
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("malicious.jpg", fake_img, "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "INVALID_MAGIC_BYTES"


def create_test_video_bytes() -> bytes:
    import tempfile
    import os
    import cv2
    import numpy as np
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_name, fourcc, 10.0, (64, 64))
        for _ in range(5):
            out.write(np.zeros((64, 64, 3), dtype=np.uint8))
        out.release()
        with open(tmp_name, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_analyze_video_async_job_and_polling():
    valid_mp4 = create_test_video_bytes()
    post_res = client.post(
        "/api/v1/analyze/video",
        files={"file": ("sample.mp4", valid_mp4, "video/mp4")}
    )
    assert post_res.status_code == 202
    job_data = post_res.json()
    assert job_data["status"] == "processing"
    assert "job_id" in job_data
    assert "poll_url" in job_data

    job_id = job_data["job_id"]

    # Poll status
    poll_res = client.get(f"/api/v1/analyze/video/jobs/{job_id}")
    assert poll_res.status_code == 200
    poll_data = poll_res.json()
    assert poll_data["job_id"] == job_id
    assert poll_data["status"] in ["processing", "completed", "blocked"]


def test_analyze_video_not_found():
    response = client.get("/api/v1/analyze/video/jobs/non-existent-uuid")
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "JOB_NOT_FOUND"
