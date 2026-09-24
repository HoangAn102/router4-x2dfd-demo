import pytest
from web.backend.app.schemas.image import FinalResult
from web.backend.app.schemas.video import VideoFinalResult


def test_image_final_result_explanation_null():
    """Verify that when explanation is not provided, it defaults to None (serializes to null)."""
    res = FinalResult(
        verdict="REAL",
        fake_probability=0.12,
        real_probability=0.88,
        continuous_score=0.12,
    )
    assert res.explanation is None
    dumped = res.model_dump()
    assert dumped["explanation"] is None


def test_image_final_result_explanation_string():
    """Verify that when explanation is provided as a string, it is preserved accurately."""
    explanation_text = "Detailed frequency analysis indicates subtle periodic pattern anomalies in high-frequency bands."
    res = FinalResult(
        verdict="FAKE",
        fake_probability=0.91,
        real_probability=0.09,
        continuous_score=0.91,
        explanation=explanation_text,
    )
    assert res.explanation == explanation_text
    dumped = res.model_dump()
    assert dumped["explanation"] == explanation_text


def test_video_final_result_explanation_null():
    """Verify that video-level explanation defaults to None."""
    res = VideoFinalResult(
        verdict="FAKE",
        fake_probability=0.88,
        real_probability=0.12,
        continuous_score=0.88,
        aggregation_method="verified_source_protocol",
    )
    assert res.explanation is None
    dumped = res.model_dump()
    assert dumped["explanation"] is None
