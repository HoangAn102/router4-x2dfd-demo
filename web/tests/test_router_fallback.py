"""PREREG_E2 P1: low-confidence routes fall back to Blending only when ROUTER_FALLBACK_POLICY=low_conf_blending."""
from unittest import mock

import pytest
import torch
from PIL import Image

from web.backend.app.config import settings
from web.model_worker.router_client import RouterClient


def _client(logits):
    rc = RouterClient.__new__(RouterClient)
    rc.device = "cpu"
    rc.model = mock.Mock(return_value=torch.tensor([logits]))
    rc._transform = lambda im: torch.zeros(3, 4, 4)
    rc.load_model = lambda: None
    return rc


@pytest.mark.parametrize("policy,logits,expected", [
    ("none", [0.0, 0.2, 0.1, 0.0], "diffusion"),                 # conf ~0.27 < 0.30, policy off
    ("low_conf_blending", [0.0, 0.2, 0.1, 0.0], "blending"),     # conf < 0.30 -> blending
    ("low_conf_blending", [0.0, 5.0, 0.0, 0.0], "diffusion"),    # confident route kept
    ("low_conf_blending", [0.3, 0.2, 0.1, 0.0], "blending"),     # low-conf blending stays blending
])
def test_fallback(tmp_path, monkeypatch, policy, logits, expected):
    img = tmp_path / "f.png"
    Image.new("RGB", (8, 8)).save(img)
    monkeypatch.setattr(settings, "ROUTER_FALLBACK_POLICY", policy)
    rc = _client(logits)
    expert, alias, conf, _ = rc.route_image(img)
    assert expert == expected and alias == expected.title()
    assert conf == round(float(torch.softmax(torch.tensor(logits), 0).max()), 4)  # confidence unchanged
    assert rc.last_route_diagnostics["fallback_applied"] == (expected != rc.last_route_diagnostics["selected_expert"])


def test_default_policy_is_off():
    assert settings.ROUTER_FALLBACK_POLICY == "none"
