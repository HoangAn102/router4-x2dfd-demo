"""The REAL/FAKE cut on the LLaVA fake probability comes from settings.DECISION_THRESHOLD (default 0.5)."""
import asyncio

from PIL import Image

from web.backend.app.config import settings
from web.model_worker.x2dfd_client import X2DFDClient


def _verdict(tmp_path, monkeypatch, threshold, fake):
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64)).save(img)
    monkeypatch.setattr(settings, "DECISION_THRESHOLD", threshold)
    c = X2DFDClient()

    async def fake_request(payload):
        return {"ok": True, "real_score": 1 - fake, "fake_score": fake}

    monkeypatch.setattr(c, "_request", fake_request)
    r = asyncio.run(c.infer(img, "blending", 0.42, include_explanation=False))
    return r.verdict, r.decision_threshold


def test_default_threshold_is_half(tmp_path, monkeypatch):
    assert settings.DECISION_THRESHOLD == 0.5
    assert _verdict(tmp_path, monkeypatch, 0.5, 0.5) == ("FAKE", 0.5)
    assert _verdict(tmp_path, monkeypatch, 0.5, 0.4999) == ("REAL", 0.5)


def test_threshold_from_settings(tmp_path, monkeypatch):
    assert _verdict(tmp_path, monkeypatch, 0.75, 0.7) == ("REAL", 0.75)
    assert _verdict(tmp_path, monkeypatch, 0.75, 0.75) == ("FAKE", 0.75)
