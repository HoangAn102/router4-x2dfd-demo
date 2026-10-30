"""Every user-facing failure returns a specific code and a Vietnamese message (FIX_PLAN PHASE 3.6)."""
import io
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

from web.backend.app.config import settings
from web.backend.app.main import app
from web.backend.app.utils.subprocess_runner import SubprocessTimeoutError

client = TestClient(app)


def _png(size=(256, 256)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def _post(data, name="x.png"):
    return client.post("/api/v1/analyze/image", files={"file": (name, data, "image/png")})


def test_corrupted_image():
    r = _post(_png()[:60])
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "CORRUPTED_IMAGE"
    assert "Ảnh bị hỏng" in r.json()["error"]["message"]


def test_unsupported_format():
    r = _post(b"GIF89a" + b"\0" * 100, name="x.gif")
    assert r.json()["error"]["code"] == "INVALID_MAGIC_BYTES"
    assert "không được hỗ trợ" in r.json()["error"]["message"]


def test_empty_file():
    r = _post(b"")
    assert r.json()["error"]["code"] == "EMPTY_FILE"


def test_too_small_image():
    r = _post(_png((40, 40)))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "IMAGE_TOO_SMALL"
    assert "quá nhỏ" in r.json()["error"]["message"]


def test_no_face(monkeypatch):
    monkeypatch.setattr(settings, "FACE_CROP_BACKEND", "haar")
    r = _post(_png())
    assert r.status_code == 400
    err = r.json()["error"]
    assert (err["code"], err["stage"]) == ("NO_FACE_DETECTED", "face_detection")
    assert "Không tìm thấy khuôn mặt" in err["message"]


def test_worker_timeout_is_504(monkeypatch):
    monkeypatch.setattr(settings, "FACE_CROP_ENABLED", False)
    worker = mock.Mock()
    worker.analyze_image = mock.AsyncMock(side_effect=SubprocessTimeoutError("expert timed out after 600s"))
    with mock.patch("web.backend.app.routers.analyze_image.get_model_worker", return_value=worker):
        r = _post(_png())
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "INFERENCE_TIMEOUT"
    assert "quá thời gian" in r.json()["error"]["message"]


def test_x2dfd_worker_timeout_message_is_504(monkeypatch):
    from web.model_worker.x2dfd_client import FinalScoreUnavailableError
    monkeypatch.setattr(settings, "FACE_CROP_ENABLED", False)
    worker = mock.Mock()
    worker.analyze_image = mock.AsyncMock(side_effect=FinalScoreUnavailableError("X2DFD persistent worker timed out"))
    with mock.patch("web.backend.app.routers.analyze_image.get_model_worker", return_value=worker):
        r = _post(_png())
    assert (r.status_code, r.json()["error"]["code"]) == (504, "INFERENCE_TIMEOUT")


def test_other_inference_error_is_500(monkeypatch):
    monkeypatch.setattr(settings, "FACE_CROP_ENABLED", False)
    worker = mock.Mock()
    worker.analyze_image = mock.AsyncMock(side_effect=RuntimeError("boom"))
    with mock.patch("web.backend.app.routers.analyze_image.get_model_worker", return_value=worker):
        r = _post(_png())
    assert (r.status_code, r.json()["error"]["code"]) == (500, "INFERENCE_FAILED")
    assert "boom" in r.json()["error"]["details"]
