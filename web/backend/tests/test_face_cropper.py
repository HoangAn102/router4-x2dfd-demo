import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from web.backend.app.config import settings
from web.backend.app.main import app
from web.backend.app.services.face_cropper import (
    NO_FACE_DETECTED_CODE,
    FaceCropper,
)
from web.backend.app.services.media_validator import (
    MediaValidationError,
    MediaValidator,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

FACE_IMAGE = (
    REPO_ROOT
    / "source_snapshot"
    / "projects"
    / "Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild"
    / "face.PNG"
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def face_crop_defaults(monkeypatch):
    # Other tests mutate the shared settings object; pin mock mode here.
    monkeypatch.setattr(settings, "RUN_MODE", "mock")
    monkeypatch.setattr(settings, "FACE_CROP_ENABLED", True)
    monkeypatch.setattr(settings, "FACE_CROP_SCALE", 1.3)
    monkeypatch.setattr(settings, "FACE_CROP_SIZE", 256)
    monkeypatch.setattr(settings, "FACE_MIN_SIZE", 60)


def _blank_jpeg_bytes(size=(320, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def test_face_image_is_cropped_to_png_256(tmp_path):
    if not FACE_IMAGE.exists():
        pytest.skip(f"Sample face image missing: {FACE_IMAGE}")

    dst = tmp_path / "face.face.png"
    out = FaceCropper.crop_face_file(FACE_IMAGE, dst)

    assert out == dst
    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.size == (256, 256)


def test_square_box_stays_inside_image():
    # Face near the top-left corner: the square is shifted, not cut.
    left, top, right, bottom = FaceCropper._square_box((0, 0, 100, 80), 300, 200)
    assert (left, top) == (0, 0)
    assert right - left == bottom - top == 130

    # Requested side larger than the image: clamped to the short side.
    left, top, right, bottom = FaceCropper._square_box((50, 20, 180, 180), 300, 200)
    assert right - left == bottom - top == 200
    assert 0 <= left and right <= 300 and top == 0 and bottom == 200


def test_blank_image_raises_no_face_detected(tmp_path):
    src = tmp_path / "blank.png"
    Image.new("RGB", (320, 240), (255, 255, 255)).save(src, format="PNG")

    with pytest.raises(MediaValidationError) as exc:
        FaceCropper.crop_face_file(src, tmp_path / "blank.face.png")

    assert exc.value.code == NO_FACE_DETECTED_CODE
    assert "Không tìm thấy khuôn mặt" in exc.value.message
    assert not (tmp_path / "blank.face.png").exists()


def test_prepare_applies_exif_orientation(tmp_path):
    path = tmp_path / "rotated.jpg"

    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW
    Image.new("RGB", (320, 240), (10, 120, 200)).save(path, format="JPEG", exif=exif)

    out_path, width, height = MediaValidator.prepare_image_file(path)

    assert (width, height) == (240, 320)
    assert out_path.suffix == ".png"
    with Image.open(out_path) as img:
        assert img.format == "PNG"
        assert img.size == (240, 320)


def test_api_image_without_face_returns_400():
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("blank.jpg", _blank_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "NO_FACE_DETECTED"
    assert data["error"]["stage"] == "face_detection"


def test_api_image_with_face_reports_face_crop_provenance():
    if not FACE_IMAGE.exists():
        pytest.skip(f"Sample face image missing: {FACE_IMAGE}")

    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("face.png", FACE_IMAGE.read_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["provenance"]["face_crop"] == "haar_1.3x_256"


def test_api_video_without_face_fails_job():
    import os
    import tempfile

    import cv2
    import numpy as np

    fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        out = cv2.VideoWriter(tmp_name, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
        for _ in range(5):
            out.write(np.zeros((64, 64, 3), dtype=np.uint8))
        out.release()
        video_bytes = Path(tmp_name).read_bytes()
    finally:
        os.remove(tmp_name)

    post_res = client.post(
        "/api/v1/analyze/video",
        files={"file": ("blank.mp4", video_bytes, "video/mp4")},
    )
    assert post_res.status_code == 202

    # TestClient runs background tasks before returning the response.
    job_id = post_res.json()["job_id"]
    poll = client.get(f"/api/v1/analyze/video/jobs/{job_id}").json()

    assert poll["status"] == "failed"
    assert poll["error"]["code"] == "NO_FACE_DETECTED"
