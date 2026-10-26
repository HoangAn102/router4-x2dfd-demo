import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from web.backend.app.config import settings
from web.backend.app.main import app
from web.backend.app.services.reliability import assess_reliability


@pytest.fixture(autouse=True)
def thresholds(monkeypatch):
    monkeypatch.setattr(settings, "RUN_MODE", "mock")
    monkeypatch.setattr(settings, "ROUTER_LOW_CONFIDENCE", 0.38)
    monkeypatch.setattr(settings, "EXPERT_NEUTRAL_BAND", 0.10)


def test_confident_router_and_informative_expert_is_normal():
    r = assess_reliability(router_confidence=0.90, calibrated_score=0.95)
    assert r.level == "normal"
    assert r.reasons == []


def test_router_confidence_boundary():
    # Exactly at the threshold is NOT low; just below is.
    assert assess_reliability(0.38, 0.95).level == "normal"
    low = assess_reliability(0.3799, 0.95)
    assert low.level == "low"
    assert len(low.reasons) == 1
    assert "Router" in low.reasons[0]


def test_neutral_expert_band_boundaries():
    # |score - 0.5| < 0.10 -> neutral; 0.40 and 0.60 sit on the edge (not neutral).
    assert assess_reliability(0.9, 0.40).level == "normal"
    assert assess_reliability(0.9, 0.60).level == "normal"
    assert assess_reliability(0.9, 0.41).level == "low"
    assert assess_reliability(0.9, 0.59).level == "low"


def test_real_case_from_live_server_is_low_with_two_reasons():
    # Live result 2026-10-26: AI image, Router texture conf=0.3074, calibrated=0.5620.
    r = assess_reliability(0.3074, 0.5620)
    assert r.level == "low"
    assert len(r.reasons) == 2
    assert "30.7%" in r.reasons[0]
    assert "0.56" in r.reasons[1]


def test_api_returns_reliability_and_keeps_verdict_fields(monkeypatch):
    monkeypatch.setattr(settings, "FACE_CROP_ENABLED", False)
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (10, 120, 200)).save(buf, format="PNG")

    resp = TestClient(app).post(
        "/api/v1/analyze/image",
        files={"file": ("t.png", buf.getvalue(), "image/png")},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["reliability"]["level"] in ("normal", "low")
    assert isinstance(data["reliability"]["reasons"], list)
    # Flag is presentation-only: the verdict contract is unchanged.
    assert data["final"]["verdict"] in ("REAL", "FAKE")
    assert data["final"]["decision_threshold"] == 0.5
