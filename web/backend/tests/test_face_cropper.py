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


# ---- detector fallback: haar -> yunet -> dlib -> NO_FACE ----

def _boom(*_a, **_k):
    raise AssertionError("fallback must not run when strict Haar found a face")


def _scene(tmp_path, size=(400, 300)):
    src = tmp_path / "scene.png"
    Image.new("RGB", size, (90, 120, 150)).save(src, format="PNG")
    return src


def test_strict_haar_hit_never_calls_fallbacks(monkeypatch):
    # Box >= 10% of the short side: no sanity re-check either.
    monkeypatch.setattr(FaceCropper, "detect_largest_face", classmethod(lambda cls, img: (10, 10, 80, 80)))
    monkeypatch.setattr(FaceCropper, "_detect_yunet", classmethod(_boom))
    monkeypatch.setattr(FaceCropper, "_detect_dlib", staticmethod(_boom))
    assert FaceCropper.detect_face_with_fallback(Image.new("RGB", (200, 200))) == ((10, 10, 80, 80), "haar")


def test_real_face_sample_still_uses_strict_haar():
    if not FACE_IMAGE.exists():
        pytest.skip(f"Sample face image missing: {FACE_IMAGE}")
    with Image.open(FACE_IMAGE) as img:
        assert FaceCropper.detect_face_with_fallback(img)[1] == "haar"


def test_yunet_used_when_haar_misses(monkeypatch, tmp_path):
    monkeypatch.setattr(FaceCropper, "detect_largest_face", classmethod(lambda cls, img: None))
    monkeypatch.setattr(FaceCropper, "_detect_yunet", classmethod(lambda cls, img: (100, 50, 120, 120)))
    monkeypatch.setattr(FaceCropper, "_detect_dlib", staticmethod(_boom))
    out, detector = FaceCropper.crop_face_file_with_detector(_scene(tmp_path), tmp_path / "s.face.png")
    assert detector == "yunet"
    assert Image.open(out).size == (256, 256)
    assert FaceCropper.provenance_tag(detector) == "yunet_1.3x_256"


def test_dlib_used_when_haar_and_yunet_miss(monkeypatch):
    monkeypatch.setattr(FaceCropper, "detect_largest_face", classmethod(lambda cls, img: None))
    monkeypatch.setattr(FaceCropper, "_detect_yunet", classmethod(lambda cls, img: None))
    monkeypatch.setattr(FaceCropper, "_detect_dlib", staticmethod(lambda img: (5, 5, 90, 90)))
    assert FaceCropper.detect_face_with_fallback(Image.new("RGB", (200, 200))) == ((5, 5, 90, 90), "dlib")


def test_all_detectors_miss_raises_no_face(monkeypatch, tmp_path):
    monkeypatch.setattr(FaceCropper, "detect_largest_face", classmethod(lambda cls, img: None))
    monkeypatch.setattr(FaceCropper, "_detect_yunet", classmethod(lambda cls, img: None))
    monkeypatch.setattr(FaceCropper, "_detect_dlib", staticmethod(lambda img: None))
    with pytest.raises(MediaValidationError) as exc:
        FaceCropper.crop_face_file_with_detector(_scene(tmp_path), tmp_path / "s.face.png")
    assert exc.value.code == NO_FACE_DETECTED_CODE


def test_yunet_missing_model_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "FACE_YUNET_MODEL", "/nonexistent/yunet.onnx")
    assert FaceCropper._detect_yunet(Image.new("RGB", (200, 200))) is None


def test_yunet_finds_nothing_on_flat_image():
    if not Path(settings.FACE_YUNET_MODEL).is_file():
        pytest.skip("YuNet model not on this machine")
    assert FaceCropper._detect_yunet(Image.new("RGB", (640, 480), (90, 120, 150))) is None


def test_video_crop_keeps_strict_haar_only(monkeypatch):
    monkeypatch.setattr(FaceCropper, "detect_largest_face", classmethod(lambda cls, img: None))
    monkeypatch.setattr(FaceCropper, "_detect_yunet", classmethod(_boom))
    assert FaceCropper.crop_image(Image.new("RGB", (200, 200))) is None


def test_api_provenance_reports_fallback_detector(monkeypatch):
    monkeypatch.setattr(settings, "RUN_MODE", "mock")
    monkeypatch.setattr(FaceCropper, "detect_largest_face", classmethod(lambda cls, img: None))
    monkeypatch.setattr(FaceCropper, "_detect_yunet", classmethod(lambda cls, img: (60, 40, 120, 120)))
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (90, 120, 150)).save(buf, format="PNG")
    resp = client.post("/api/v1/analyze/image", files={"file": ("selfie.png", buf.getvalue(), "image/png")})
    assert resp.status_code == 200
    assert resp.json()["provenance"]["face_crop"] == "yunet_1.3x_256"


def test_tiny_haar_box_is_replaced_by_larger_yunet_face(monkeypatch):
    # 70px Haar box on a 1920x2560 photo (< 10% of short side) -> YuNet's real face wins.
    monkeypatch.setattr(FaceCropper, "detect_largest_face", classmethod(lambda cls, img: (1332, 687, 70, 70)))
    monkeypatch.setattr(FaceCropper, "_detect_yunet", classmethod(lambda cls, img: (579, 542, 634, 821)))
    assert FaceCropper.detect_face_with_fallback(Image.new("RGB", (1920, 2560))) == ((579, 542, 634, 821), "yunet")


def test_tiny_haar_box_kept_when_yunet_finds_nothing(monkeypatch):
    monkeypatch.setattr(FaceCropper, "detect_largest_face", classmethod(lambda cls, img: (1332, 687, 70, 70)))
    monkeypatch.setattr(FaceCropper, "_detect_yunet", classmethod(lambda cls, img: None))
    assert FaceCropper.detect_face_with_fallback(Image.new("RGB", (1920, 2560))) == ((1332, 687, 70, 70), "haar")
