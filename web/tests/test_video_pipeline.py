import io
from pathlib import Path
from PIL import Image
from web.backend.app.schemas.video import FrameResult
from web.backend.app.services.video_aggregator import VideoAggregator
from web.backend.app.services.video_extractor import VideoExtractor


def test_video_extractor_simulated_frames(tmp_path):
    import cv2
    import numpy as np

    video_dummy = tmp_path / "test_dummy.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_dummy), fourcc, 10.0, (64, 64))
    for _ in range(5):
        out.write(np.zeros((64, 64, 3), dtype=np.uint8))
    out.release()

    out_dir = tmp_path / "frames_out"
    frames = VideoExtractor.extract_uniform_frames(video_dummy, out_dir, target_num_frames=8)

    assert len(frames) > 0
    assert frames[0]["frame_index"] == 0
    assert "timestamp_seconds" in frames[0]
    assert Path(frames[0]["frame_path"]).exists()
    assert frames[0]["thumbnail_b64"] is not None


def test_video_aggregator_unverified_returns_none_and_blocked():
    """Verify fail-closed behavior when aggregation algorithm is unverified."""
    agg = VideoAggregator(verified_protocol=None)
    mock_frames = [
        FrameResult(
            frame_index=0,
            timestamp_seconds=0.0,
            selected_expert="frequency",
            expert_score=0.8,
            final_frame_score=0.85,
            verdict="FAKE",
        ),
        FrameResult(
            frame_index=1,
            timestamp_seconds=0.4,
            selected_expert="blending",
            expert_score=0.7,
            final_frame_score=0.75,
            verdict="FAKE",
        ),
    ]

    final_res, error_code, error_msg = agg.aggregate(mock_frames)
    assert final_res is None
    assert error_code == "VIDEO_AGGREGATION_UNVERIFIED"
    assert "unverified" in error_msg.lower()


def test_video_aggregator_verified_mean_protocol():
    """Verify aggregation calculation when verified protocol is provided."""
    agg = VideoAggregator(verified_protocol="mean_probability")
    mock_frames = [
        FrameResult(
            frame_index=0,
            timestamp_seconds=0.0,
            selected_expert="frequency",
            expert_score=0.8,
            final_frame_score=0.8,
            verdict="FAKE",
        ),
        FrameResult(
            frame_index=1,
            timestamp_seconds=0.4,
            selected_expert="blending",
            expert_score=0.6,
            final_frame_score=0.6,
            verdict="FAKE",
        ),
    ]

    final_res, error_code, error_msg = agg.aggregate(mock_frames)
    assert final_res is not None
    assert error_code is None
    assert final_res.verdict == "FAKE"
    assert final_res.fake_probability == 0.7  # (0.8 + 0.6) / 2
    assert final_res.real_probability == 0.3
    assert final_res.continuous_score == 0.7
