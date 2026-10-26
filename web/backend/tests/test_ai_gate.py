import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from web.backend.app.config import settings
from web.backend.app.main import app
from web.backend.app.schemas.image import AIDetectorResult, FinalResult, Reliability
from web.backend.app.services.ai_gate import apply_ai_gate, build_faceless_response
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


def test_build_faceless_response_threshold():
    assert build_faceless_response("r", None, {}, {}) is None
    assert build_faceless_response("r", 0.9899, {}, {}) is None
    resp = build_faceless_response("r", 0.99, {"total": 1.0}, {"mode": "m"})
    assert resp.final.verdict == "FAKE"
    assert resp.router is None and resp.expert is None
    assert resp.ai_detector.overridden is True
    assert "không áp dụng" in resp.final.explanation


def _post_faceless():
    # A flat colour image: Haar finds no face -> NO_FACE_DETECTED path.
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), (10, 120, 200)).save(buf, format="PNG")
    return TestClient(app).post(
        "/api/v1/analyze/image",
        files={"file": ("scene.png", buf.getvalue(), "image/png")},
    )


def test_faceless_ai_image_returns_fake_from_gate(monkeypatch):
    monkeypatch.setattr(settings, "FACE_CROP_ENABLED", True)
    monkeypatch.setattr(MockModelWorker, "_mock_ai_score", lambda self, b: 0.997)
    resp = _post_faceless()
    assert resp.status_code == 200
    data = resp.json()
    assert data["final"]["verdict"] == "FAKE"
    assert data["final"]["fake_probability"] == pytest.approx(0.997)
    assert data["router"] is None
    assert data["expert"] is None
    assert data["ai_detector"]["overridden"] is True
    assert data["ai_detector"]["x2dfd_fake_probability"] is None
    assert data["provenance"]["face_crop"] == "no_face"
    assert data["provenance"]["pipeline"] == "ai_gate_only"


@pytest.mark.parametrize("enabled, score", [(True, 0.5), (True, 0.9899), (False, 0.999)])
def test_faceless_below_threshold_or_disabled_keeps_no_face_error(monkeypatch, enabled, score):
    monkeypatch.setattr(settings, "FACE_CROP_ENABLED", True)
    monkeypatch.setattr(settings, "AI_GATE_ENABLED", enabled)
    original = MockModelWorker._mock_ai_score
    # Disabled gate: the real mock hook returns None regardless of the patched score.
    monkeypatch.setattr(
        MockModelWorker, "_mock_ai_score",
        (lambda self, b: score) if enabled else original,
    )
    resp = _post_faceless()
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "NO_FACE_DETECTED"


# ---- blending gate + shared decide() ----

from web.backend.app.schemas.image import BlendingGateResult  # noqa: E402
from web.backend.app.services.decision import decide  # noqa: E402


@pytest.fixture
def blending_settings(monkeypatch):
    monkeypatch.setattr(settings, "BLENDING_GATE_ENABLED", True)
    monkeypatch.setattr(settings, "BLENDING_GATE_THRESHOLD", 0.7153)


def test_decide_without_scores_is_identity(blending_settings):
    final, rel = x2dfd_real(), Reliability()
    out_final, out_rel, ai, bl = decide(final, rel, None, None)
    assert out_final == final and out_rel == rel and ai is None and bl is None


def test_blending_at_threshold_overrides_real_verdict(blending_settings):
    final, _, ai, bl = decide(x2dfd_real(0.02), Reliability(), 0.05, 0.7153)
    assert final.verdict == "FAKE"
    assert final.fake_probability == pytest.approx(0.7153)
    assert final.real_probability == pytest.approx(1 - 0.7153)
    assert final.explanation.startswith("Phát hiện dấu vết ghép/hòa trộn khuôn mặt (expert blending), điểm 0.715")
    assert bl.overridden is True and bl.x2dfd_fake_probability == pytest.approx(0.02)
    assert ai.overridden is False


def test_blending_below_threshold_keeps_x2dfd(blending_settings):
    final, _, _, bl = decide(x2dfd_real(), Reliability(), 0.05, 0.7152)
    assert final == x2dfd_real()
    assert bl.overridden is False and bl.raw == pytest.approx(0.7152)


def test_blending_does_not_replace_an_x2dfd_fake(blending_settings):
    fake = FinalResult(verdict="FAKE", fake_probability=0.86, real_probability=0.14, continuous_score=0.86, explanation="x2dfd text")
    final, _, _, bl = decide(fake, Reliability(), 0.05, 0.99)
    assert final == fake and bl.overridden is False


def test_ai_gate_takes_precedence_over_blending(blending_settings):
    final, _, ai, bl = decide(x2dfd_real(), Reliability(), 0.999, 0.95)
    assert ai.overridden is True and bl.overridden is False
    assert final.fake_probability == pytest.approx(0.999)
    assert final.explanation.startswith("Phát hiện dấu hiệu ảnh do AI tạo")


def test_blending_gate_disabled_reports_nothing(monkeypatch, blending_settings):
    monkeypatch.setattr(settings, "BLENDING_GATE_ENABLED", False)
    final, _, _, bl = decide(x2dfd_real(), Reliability(), 0.05, 0.99)
    assert final == x2dfd_real() and bl is None


def test_blending_threshold_is_read_from_settings(monkeypatch, blending_settings):
    monkeypatch.setattr(settings, "BLENDING_GATE_THRESHOLD", 0.9)
    final, _, _, bl = decide(x2dfd_real(), Reliability(), 0.05, 0.8)
    assert final.verdict == "REAL" and bl.threshold == pytest.approx(0.9)


def test_blending_schema_rejects_out_of_range():
    with pytest.raises(ValidationError):
        BlendingGateResult(raw=1.5, threshold=0.7)


def test_api_blending_override_and_disabled(monkeypatch, blending_settings):
    monkeypatch.setattr(MockModelWorker, "_mock_ai_score", lambda self, b: 0.01)
    monkeypatch.setattr(MockModelWorker, "_mock_blending_raw", lambda self, b: 0.99)
    # Mock X2DFD verdict depends on the file hash; force it REAL via a tiny wrapper on decide input.
    import web.model_worker.mock_worker as mw
    real_decide = mw.decide
    monkeypatch.setattr(mw, "decide", lambda f, r, a, b: real_decide(f.model_copy(update={"verdict": "REAL", "fake_probability": 0.1, "real_probability": 0.9, "continuous_score": 0.1}), r, a, b))
    data = _post_png().json()
    assert data["final"]["verdict"] == "FAKE"
    assert data["blending_gate"]["overridden"] is True
    assert data["blending_gate"]["x2dfd_fake_probability"] == pytest.approx(0.1)
    monkeypatch.setattr(settings, "BLENDING_GATE_ENABLED", False)
    data = _post_png().json()
    assert data["blending_gate"] is None and data["final"]["verdict"] == "REAL"


def test_api_schema_contains_blending_gate_fields(blending_settings):
    bl = _post_png().json()["blending_gate"]
    assert set(bl) == {"raw", "threshold", "overridden", "x2dfd_fake_probability"}
    assert bl["overridden"] is False
