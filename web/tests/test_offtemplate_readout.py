"""Off-template LLaVA answers (R6 31/10: V3 LoRA answered "ant") are scored by teacher forcing and flagged;
they never raise, and the API returns 200 with a low-reliability note."""
import asyncio
import io
import json
import math
import sys
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

from web.backend.app.config import settings
from web.backend.app.main import app
from web.backend.app.schemas.image import ExpertResult, FinalResult, ImageAnalyzeResponse, RouterResult
from web.backend.app.services.reliability import assess_reliability
from web.model_worker.x2dfd_client import X2DFDClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_worker"))
import x2dfd_worker_cli as cli  # noqa: E402

PREFIX = [910, 1967, 338]  # "This image is" (LLaMA tokenizer)
REAL, FAKE = 1855, 25713


def test_forced_readout_on_template():
    r = cli.forced_readout(PREFIX, PREFIX, FAKE, 1.0, 3.0, REAL, FAKE)
    assert r["on_template"] is True
    assert math.isclose(r["fake_score"], 1 / (1 + math.exp(-2.0)), abs_tol=1e-8)
    assert math.isclose(r["real_score"] + r["fake_score"], 1.0, abs_tol=1e-8)


def test_forced_readout_off_template_prefix_is_flagged_not_raised():
    # greedy would start with "ant" instead of "This"
    r = cli.forced_readout(PREFIX, [424, 1967, 338], REAL, 2.0, -1.0, REAL, FAKE)
    assert r["on_template"] is False
    assert r["fake_score"] < 0.5


def test_forced_readout_off_template_label():
    r = cli.forced_readout(PREFIX, PREFIX, 263, 0.0, 0.0, REAL, FAKE)  # label slot argmax is neither real/fake
    assert (r["on_template"], r["fake_score"]) == (False, 0.5)


def test_forced_readout_extreme_logits_do_not_overflow():
    assert cli.forced_readout(PREFIX, PREFIX, FAKE, -1e4, 1e4, REAL, FAKE)["fake_score"] == 1.0
    assert cli.forced_readout(PREFIX, PREFIX, REAL, 1e4, -1e4, REAL, FAKE)["fake_score"] == 0.0


def test_worker_run_one_passes_flag(tmp_path):
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64)).save(img)
    score = mock.Mock(return_value={"answer": "This image is real", "label_step": 3, "real_score": 0.8,
                                    "fake_score": 0.2, "on_template": False})
    out = cli._run_one(score, mock.Mock(), mode="score", image=str(img), prompt="p", lora_dir="L",
                       base_model="B", max_new_tokens=4)
    assert out["ok"] is True and out["on_template"] is False and out["fake_score"] == 0.2


def test_client_keeps_flag_and_verdict(tmp_path, monkeypatch):
    img = tmp_path / "f.png"
    Image.new("RGB", (64, 64)).save(img)
    c = X2DFDClient()

    async def fake_request(payload):
        return {"ok": True, "real_score": 0.3, "fake_score": 0.7, "on_template": False}

    monkeypatch.setattr(c, "_request", fake_request)
    r = asyncio.run(c.infer(img, "blending", 0.42, include_explanation=False))
    assert (r.verdict, r.on_template) == ("FAKE", False)


def test_reliability_reason():
    low = assess_reliability(0.9, 0.95, llava_on_template=False)
    assert low.level == "low" and "khuôn" in low.reasons[0]
    assert assess_reliability(0.9, 0.95).level == "normal"


def test_api_returns_200_for_offtemplate(monkeypatch):
    monkeypatch.setattr(settings, "FACE_CROP_ENABLED", False)
    resp = ImageAnalyzeResponse(
        request_id="t",
        router=RouterResult(selected_expert="blending", selected_alias="Blending", router_confidence=0.9, margin=0.5),
        expert=ExpertResult(name="blending", raw_score=0.2, calibrated_score=0.32),
        final=FinalResult(verdict="REAL", fake_probability=0.1, real_probability=0.9, continuous_score=0.1,
                          on_template=False),
        reliability=assess_reliability(0.9, 0.32, llava_on_template=False),
        timing_ms={}, provenance={},
    )
    worker = mock.Mock()
    worker.analyze_image = mock.AsyncMock(return_value=resp)
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (120, 120, 120)).save(buf, format="PNG")
    with mock.patch("web.backend.app.routers.analyze_image.get_model_worker", return_value=worker):
        r = TestClient(app).post("/api/v1/analyze/image", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["final"]["on_template"] is False and body["reliability"]["level"] == "low"
    json.dumps(body)
