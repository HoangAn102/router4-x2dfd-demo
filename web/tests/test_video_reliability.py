"""Video frames carry the same presentation-only reliability flag as images (verdict unchanged)."""
import asyncio
from unittest import mock

from web.backend.app.schemas.image import FinalResult
from web.backend.app.schemas.video import FrameResult
from web.model_worker.live_worker import LiveModelWorker


def _worker(router_conf):
    w = LiveModelWorker.__new__(LiveModelWorker)
    w.router_client = mock.Mock()
    w.router_client.route_image.return_value = ("blending", "Blending", router_conf, 0.1)
    w.expert_client = mock.Mock()
    w.expert_client.compute_expert_score = mock.AsyncMock(return_value=(0.9, 0.8))
    w.x2dfd_client = mock.Mock()
    w.x2dfd_client.infer = mock.AsyncMock(return_value=FinalResult(
        verdict="FAKE", fake_probability=0.9, real_probability=0.1, continuous_score=0.9,
        decision_threshold=0.5, explanation=None))
    return w


def test_low_router_confidence_flags_frame_without_changing_verdict(tmp_path):
    f = tmp_path / "f.png"
    f.touch()
    with mock.patch("web.model_worker.live_worker.GpuLock"):
        low = asyncio.run(_worker(0.25).analyze_video_frame(f, 0, 0.0))
        ok = asyncio.run(_worker(0.60).analyze_video_frame(f, 0, 0.0))
    assert (low.reliability.level, low.verdict, low.final_frame_score) == ("low", "FAKE", 0.9)
    assert ok.reliability.level == "normal"


def test_frame_result_default_reliability_is_normal():
    fr = FrameResult(frame_index=0, timestamp_seconds=0.0, selected_expert="texture", expert_score=0.5,
                     final_frame_score=0.2, verdict="REAL")
    assert fr.reliability.level == "normal" and fr.reliability.reasons == []
