import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from web.backend.app.config import settings
from web.backend.app.main import app
from web.backend.app.schemas.image import AIDetectorResult, FinalResult, Reliability
from web.backend.app.services.ai_gate import apply_ai_gate
from web.model_worker.ai_image_detector import AIImageDetector
from web.model_worker.mock_worker import MockModelWorker


@pytest.fixture(autouse=True)
def gate_settings(monkeypatch):
    monkeypatch.setattr(settings, "RUN_MODE", "mock")
    monkeypatch.setattr(settings, "FACE_CROP_ENABLED", False)
    monkeypatch.setattr(settings, "AI_GATE_ENABLED", True)
    monkeypatch.setattr(settings, "AI_GATE_THRESHOLD", 0.99)
    monkeypatch.setattr(settings, "AI_GATE_WARN", 0.95)


def x2dfd_real(fake=0.02):
    return FinalResult(
        verdict="REAL",
        fake_probability=fake,
        real_probability=1.0 - fake,
        continuous_score=fake,
        explanation="original X2DFD text",
    )


def test_no_score_leaves_result_untouched():
    final, rel = x2dfd_real(), Reliability()
    out_final, out_rel, det = apply_ai_gate(final, rel, None)
    assert out_final == final
    assert out_rel == rel
    assert det is None


def test_score_at_threshold_overrides_to_fake():
    final, rel, det = apply_ai_gate(x2dfd_real(0.021), Reliability(), 0.99)
    assert final.verdict == "FAKE"
    assert final.fake_probability == pytest.approx(0.99)
    assert final.real_probability == pytest.approx(0.01)
    assert final.continuous_score == pytest.approx(0.99)
    assert final.explanation.startswith("Phát hiện dấu hiệu ảnh do AI tạo (detector tổng quát), điểm 0.990")
    assert det.overridden is True
    assert det.x2dfd_fake_probability == pytest.approx(0.021)
    assert det.threshold == pytest.approx(0.99)
    assert rel.level == "normal"


def test_warn_band_keeps_verdict_but_lowers_reliability():
    prior = Reliability(level="low", reasons=["router reason"])
    final, rel, det = apply_ai_gate(x2dfd_real(), prior, 0.9899)
    assert final.verdict == "REAL"
    assert final.fake_probability == pytest.approx(0.02)
    assert det.overridden is False
    assert det.x2dfd_fake_probability is None
    assert rel.level == "low"
    assert rel.reasons[0] == "router reason"
    assert "Có dấu hiệu ảnh do AI tạo (điểm 0.990), chưa đủ chắc chắn." in rel.reasons[1]


def test_below_warn_is_reported_but_changes_nothing():
    final, rel, det = apply_ai_gate(x2dfd_real(), Reliability(), 0.9499)
    assert final == x2dfd_real()
    assert rel == Reliability()
    assert det.score == pytest.approx(0.9499)
    assert det.overridden is False


def test_threshold_is_read_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "AI_GATE_THRESHOLD", 0.995)
    final, _, det = apply_ai_gate(x2dfd_real(), Reliability(), 0.993)
    assert final.verdict == "REAL"
    assert det.overridden is False


def test_ai_detector_schema_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        AIDetectorResult(score=1.2, threshold=0.99, model="m")


def test_detector_disabled_or_missing_model_returns_none(monkeypatch, tmp_path):
    img = tmp_path / "x.png"
    Image.new("RGB", (32, 32)).save(img)
    detector = AIImageDetector(model_dir=str(tmp_path / "does-not-exist"))
    assert detector.try_score(img) is None          # missing model: fail open
    monkeypatch.setattr(settings, "AI_GATE_ENABLED", False)
    assert detector.try_score(img) is None


def _post_png():
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (10, 120, 200)).save(buf, format="PNG")
    return TestClient(app).post(
        "/api/v1/analyze/image",
        files={"file": ("t.png", buf.getvalue(), "image/png")},
    )


def test_api_override_when_gate_enabled(monkeypatch):
    monkeypatch.setattr(MockModelWorker, "_mock_ai_score", lambda self, b: 0.999)
    resp = _post_png()
    assert resp.status_code == 200
    data = resp.json()
    assert data["final"]["verdict"] == "FAKE"
    assert data["final"]["fake_probability"] == pytest.approx(0.999)
    assert data["ai_detector"]["overridden"] is True
    assert data["ai_detector"]["threshold"] == pytest.approx(0.99)
    assert data["ai_detector"]["model"] == "haywoodsloan-ai-image-detector-deploy"
    assert 0.0 <= data["ai_detector"]["x2dfd_fake_probability"] <= 1.0


def test_api_disabled_gate_is_old_behaviour(monkeypatch):
    monkeypatch.setattr(settings, "AI_GATE_ENABLED", False)
    data = _post_png().json()
    assert data["ai_detector"] is None
    # Same verdict rule as before the gate existed: X2DFD probability vs 0.5.
    fake = data["final"]["fake_probability"]
    assert data["final"]["verdict"] == ("FAKE" if fake >= 0.5 else "REAL")


def test_api_schema_contains_ai_detector_fields():
    data = _post_png().json()
    det = data["ai_detector"]
    assert set(det) == {"score", "threshold", "model", "overridden", "x2dfd_fake_probability"}
    assert det["overridden"] is False          # mock score stays below 0.5
