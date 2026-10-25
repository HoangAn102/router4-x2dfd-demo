import pytest
from web.model_worker.router_client import ALIASES, EXPERTS, RouterClient
from web.model_worker.x2dfd_client import FinalScoreUnavailableError, X2DFDClient


def test_wfs_prompt_builder():
    client = X2DFDClient()
    prompt = client.build_wfs_prompt("Frequency", 0.812)
    assert prompt == "<image>\nIs this image real or fake? And the frequency score is 0.812."

    prompt_blend = client.build_wfs_prompt("Blending", 0.054)
    assert prompt_blend == "<image>\nIs this image real or fake? And the blending score is 0.054."


def test_x2dfd_validate_scores_rejections():
    client = X2DFDClient()

    # Reject bool
    with pytest.raises(FinalScoreUnavailableError, match="cannot be bool"):
        client.validate_scores(True, 0.5)

    # Reject NaN
    with pytest.raises(FinalScoreUnavailableError, match="Non-finite"):
        client.validate_scores(0.5, float("nan"))

    # Reject out of range
    with pytest.raises(FinalScoreUnavailableError, match=r"outside \[0,1\]"):
        client.validate_scores(1.5, -0.5)

    # Reject unnormalized
    with pytest.raises(FinalScoreUnavailableError, match="not normalized"):
        client.validate_scores(0.8, 0.4)

    # Valid
    real_f, fake_f = client.validate_scores(0.2, 0.8)
    assert real_f == 0.2
    assert fake_f == 0.8


def test_router_client_constants():
    assert len(EXPERTS) == 4
    assert set(EXPERTS) == {"blending", "diffusion", "frequency", "texture"}
    assert ALIASES["blending"] == "Blending"
    assert ALIASES["diffusion"] == "Diffusion"
    assert ALIASES["frequency"] == "Frequency"
    assert ALIASES["texture"] == "Texture"
