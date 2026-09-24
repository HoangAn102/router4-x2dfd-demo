import math
import pytest
from pydantic import ValidationError
from web.backend.app.schemas.image import FinalResult
from web.backend.app.schemas.video import FrameResult


def test_final_result_valid():
    fr = FinalResult(
        verdict="FAKE",
        fake_probability=0.85,
        real_probability=0.15,
        continuous_score=0.85,
        decision_threshold=0.5,
        explanation="Detected artifacts.",
    )
    assert fr.verdict == "FAKE"
    assert fr.fake_probability == 0.85
    assert fr.real_probability == 0.15
    assert fr.explanation == "Detected artifacts."


def test_final_result_rejects_bool():
    """Python treats bool as int; our validator must strictly reject booleans."""
    with pytest.raises(ValidationError, match="Score cannot be a boolean"):
        FinalResult(
            verdict="FAKE",
            fake_probability=True,  # type: ignore
            real_probability=0.0,
            continuous_score=1.0,
        )


def test_final_result_rejects_nan():
    with pytest.raises(ValidationError, match="Score must be finite"):
        FinalResult(
            verdict="FAKE",
            fake_probability=float("nan"),
            real_probability=0.5,
            continuous_score=0.5,
        )


def test_final_result_rejects_out_of_bounds():
    with pytest.raises(ValidationError, match="Score must be in range"):
        FinalResult(
            verdict="FAKE",
            fake_probability=1.2,
            real_probability=-0.2,
            continuous_score=1.2,
        )


def test_final_result_rejects_unnormalized():
    with pytest.raises(ValidationError, match="Pairwise probabilities must sum to 1.0"):
        FinalResult(
            verdict="FAKE",
            fake_probability=0.7,
            real_probability=0.5,  # Sum = 1.2 != 1.0
            continuous_score=0.7,
        )


def test_frame_result_rejects_bool():
    with pytest.raises(ValidationError, match="Score cannot be a boolean"):
        FrameResult(
            frame_index=0,
            timestamp_seconds=0.0,
            selected_expert="frequency",
            expert_score=False,  # type: ignore
            final_frame_score=0.5,
            verdict="REAL",
        )
