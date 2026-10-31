"""R7 (2): an explanation whose first sentence contradicts the score verdict is flagged; verdict and text unchanged."""
import asyncio

import pytest
from PIL import Image

from web.model_worker.x2dfd_client import X2DFDClient


@pytest.mark.parametrize("text,label", [
    ("This image is real. The image features a woman.", "REAL"),
    ("This image is fake. As an expert deepfake detector, it looks real at first.", "FAKE"),
    ("This image is FAKE.\n1. Eyes: real-looking", "FAKE"),
    ("The image shows a man.", None),
    ("It is real or fake, hard to say.", None),
    ("", None),
    (None, None),
])
def test_explanation_label(text, label):
    assert X2DFDClient.explanation_label(text) == label


@pytest.mark.parametrize("explanation,expected", [
    ("This image is fake. Blending artifacts near the jaw.", False),
    ("This image is real. Natural skin texture.", True),
    ("The image shows a man.", None),
])
def test_client_flags_without_changing_verdict_or_text(tmp_path, monkeypatch, explanation, expected):
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64)).save(img)
    c = X2DFDClient()

    async def fake_request(payload):
        if payload["mode"] == "score":
            return {"ok": True, "real_score": 0.8, "fake_score": 0.2, "on_template": True}
        return {"ok": True, "answer": explanation}

    monkeypatch.setattr(c, "_request", fake_request)
    r = asyncio.run(c.infer(img, "blending", 0.42, include_explanation=True))
    assert r.verdict == "REAL" and r.explanation == explanation
    assert r.explanation_consistent is expected
