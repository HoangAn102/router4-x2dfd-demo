import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from web.backend.app.config import settings
from web.backend.app.services import dlib_face_backend as dfb
from web.backend.app.services.face_cropper import FaceCropper


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(settings, "FACE_CROP_BACKEND", "auto")
    monkeypatch.setattr(settings, "FACE_CROP_ENABLED", True)
    monkeypatch.setattr(settings, "FACE_CROP_SCALE", 1.3)
    monkeypatch.setattr(settings, "FACE_CROP_SIZE", 256)


@pytest.mark.parametrize("mode,backend", [("live", "auto"), ("mock", "dlib"), ("live", "dlib")])
def test_missing_dlib_fail_closed(monkeypatch, mode, backend):
    monkeypatch.setattr(settings, "RUN_MODE", mode)
    monkeypatch.setattr(settings, "FACE_CROP_BACKEND", backend)
    def missing():
        raise dfb.FaceCropBackendUnavailable("dlib missing")
    monkeypatch.setattr(dfb, "runtime", missing)
    with pytest.raises(dfb.FaceCropBackendUnavailable):
        dfb.backend()


def test_mock_missing_dlib_uses_haar(monkeypatch, caplog):
    monkeypatch.setattr(settings, "RUN_MODE", "mock")
    def missing():
        raise dfb.FaceCropBackendUnavailable("missing test dependency")
    monkeypatch.setattr(dfb, "runtime", missing)
    dfb._warn_fallback.cache_clear()
    assert dfb.backend() == "haar"
    assert "fallback" in caplog.text


def test_haar_does_not_import_dlib(monkeypatch):
    monkeypatch.setattr(settings, "FACE_CROP_BACKEND", "haar")
    monkeypatch.setattr(dfb, "runtime", lambda: pytest.fail("dlib loaded in Haar mode"))
    assert dfb.backend() == "haar"
    assert FaceCropper.provenance_tag() == "haar_1.3x_256"


def test_bgr_and_png_roundtrip(monkeypatch, tmp_path):
    import cv2
    monkeypatch.setattr(settings, "FACE_CROP_BACKEND", "dlib")
    bgr = np.zeros((256, 256, 3), dtype=np.uint8)
    bgr[:] = [13, 51, 201]
    seen = []
    def fn(detector, predictor, image):
        seen.append(image.copy())
        return image.copy(), np.zeros((81, 2)), None
    monkeypatch.setattr(dfb, "runtime", lambda: (fn, object(), object()))
    source, dest = tmp_path / "source.png", tmp_path / "face.png"
    assert cv2.imwrite(str(source), bgr)
    FaceCropper.crop_face_file(source, dest)
    np.testing.assert_array_equal(seen[0], bgr)
    np.testing.assert_array_equal(cv2.imread(str(dest)), bgr)
    assert FaceCropper.provenance_tag() == "dlib_deepfakebench_align_1.3x_256"


@pytest.mark.parametrize("result", [None, (None, None, None)])
def test_no_face_is_not_fallback(monkeypatch, result):
    monkeypatch.setattr(dfb, "runtime", lambda: (lambda *args: result, None, None))
    assert FaceCropper.crop_image(Image.new("RGB", (256, 256))) is None


def test_signature_rejected():
    with pytest.raises(dfb.FaceCropBackendUnavailable, match="signature"):
        dfb._check_contract(lambda image: image)


def test_no_face_file_preserves_error_code(monkeypatch, tmp_path):
    from web.backend.app.services.media_validator import MediaValidationError
    monkeypatch.setattr(dfb, "runtime", lambda: (lambda *args: (None, None, None), None, None))
    source, dest = tmp_path / "blank.png", tmp_path / "crop.png"
    Image.new("RGB", (256, 256)).save(source)
    with pytest.raises(MediaValidationError) as error:
        FaceCropper.crop_face_file(source, dest)
    assert error.value.code == "NO_FACE_DETECTED"
    assert not dest.exists()


def test_wrong_landmark_model_is_rejected(monkeypatch):
    result = (np.zeros((256, 256, 3), np.uint8), np.zeros((68, 2)), None)
    monkeypatch.setattr(dfb, "runtime", lambda: (lambda *args: result, None, None))
    with pytest.raises(dfb.FaceCropBackendUnavailable, match="81 landmarks"):
        FaceCropper.crop_image(Image.new("RGB", (256, 256)))


def test_cached_load_without_chdir(monkeypatch, tmp_path):
    # Stand-in source tests loader/cache mechanics, not real dlib parity.
    source = tmp_path / "preprocess.py"
    source.write_text('''def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, scale=1.3):
        return img
    return None, None, None
''')
    predictor = tmp_path / "predictor.dat"
    predictor.touch()
    imports = []
    fake_dlib = SimpleNamespace(
        get_frontal_face_detector=lambda: imports.append("detector"),
        shape_predictor=lambda path: imports.append(path))
    real_import = dfb.importlib.import_module
    monkeypatch.setattr(dfb.importlib, "import_module", lambda name: fake_dlib if name == "dlib" else real_import(name))
    dfb._load.cache_clear()
    cwd = Path.cwd()
    a = dfb._load(str(tmp_path), str(predictor))
    b = dfb._load(str(tmp_path), str(predictor))
    assert a is b
    assert Path.cwd() == cwd
    assert len(imports) == 2
    dfb._load.cache_clear()


def test_api_returns_backend_unavailable_500(monkeypatch):
    from fastapi.testclient import TestClient
    from web.backend.app.main import app
    import io
    def missing():
        raise dfb.FaceCropBackendUnavailable("required dlib unavailable")
    monkeypatch.setattr(dfb, "runtime", missing)
    monkeypatch.setattr(settings, "FACE_CROP_BACKEND", "dlib")
    data = io.BytesIO()
    Image.new("RGB", (256, 256)).save(data, format="PNG")
    response = TestClient(app).post("/api/v1/analyze/image", files={"file": ("test.png", data.getvalue(), "image/png")})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "FACE_CROP_BACKEND_UNAVAILABLE"


def test_live_readiness(monkeypatch):
    from web.model_worker.live_worker import LiveModelWorker
    monkeypatch.setattr(settings, "RUN_MODE", "live")
    def missing():
        raise dfb.FaceCropBackendUnavailable("required dlib unavailable")
    monkeypatch.setattr(dfb, "runtime", missing)
    worker = LiveModelWorker.__new__(LiveModelWorker)
    assert not worker.check_readiness()["face_cropper"].ready


def test_real_dlib_integration(monkeypatch, tmp_path):
    pytest.importorskip("dlib")
    root = Path(settings.DEEPFAKEBENCH_PREPROCESS_DIR)
    if not (root / "preprocess.py").is_file():
        pytest.skip("DeepfakeBench preprocessing not installed on this machine")
    predictor = (Path(settings.DLIB_PREDICTOR_PATH) if settings.DLIB_PREDICTOR_PATH else
                 root / "dlib_tools/shape_predictor_81_face_landmarks.dat")
    if not predictor.is_file():
        pytest.skip("81-landmark predictor not installed")
    monkeypatch.setattr(settings, "FACE_CROP_BACKEND", "dlib")
    fn, detector, pred = dfb.runtime()
    import cv2
    sample = os.environ.get("DFB_TEST_FACE_IMAGE")
    if not sample:
        sample = str(Path(__file__).resolve().parents[3] / "source_snapshot/projects/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild/face.PNG")
    if not Path(sample).is_file():
        pytest.skip("Set DFB_TEST_FACE_IMAGE to a known real face PNG on server")
    frame = cv2.imread(sample)
    assert frame is not None
    direct, landmarks, _ = fn(detector, pred, frame.copy())
    assert direct is not None and landmarks is not None, "Fixture must contain a detectable face"
    source = tmp_path / "input.png"
    cv2.imwrite(str(source), frame)
    output = tmp_path / "crop.png"
    FaceCropper.crop_face_file(source, output)
    np.testing.assert_array_equal(cv2.imread(str(output)), direct)
