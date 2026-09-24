import time
import pytest
from web.backend.app.services.job_manager import JobManager
from web.backend.app.schemas.video import VideoAnalyzeResponse, VideoInfo, FrameResult


def test_job_manager_create_and_get():
    mgr = JobManager()
    job = mgr.create_job("job_123")
    assert job.job_id == "job_123"
    assert job.status == "processing"
    assert job.stage == "queued"

    retrieved = mgr.get_job("job_123")
    assert retrieved is not None
    assert retrieved.job_id == "job_123"


def test_job_manager_progress_update():
    mgr = JobManager()
    mgr.create_job("job_progress")
    mgr.update_progress("job_progress", stage="analyzing_frames", current_frame=5, total_frames=32)

    job = mgr.get_job("job_progress")
    assert job.stage == "analyzing_frames"
    assert job.current_frame == 5
    assert job.total_frames == 32


def test_job_manager_block_and_complete():
    mgr = JobManager()
    mgr.create_job("job_block")

    dummy_resp = VideoAnalyzeResponse(
        request_id="job_block",
        status="blocked",
        media_type="video",
        video=VideoInfo(
            duration_seconds=1.0,
            fps=30.0,
            total_frames_extracted=1,
            frames_analyzed=1,
            sampling_protocol="uniform",
        ),
        final=None,
        error_code="VIDEO_AGGREGATION_UNVERIFIED",
        message="Blocked test",
        frames=[],
        timing_ms={"total": 100.0},
        provenance={"mode": "test"},
    )

    mgr.block_job("job_block", dummy_resp)
    job = mgr.get_job("job_block")
    assert job.status == "blocked"
    assert job.result.error_code == "VIDEO_AGGREGATION_UNVERIFIED"

    status_resp = job.to_status_response()
    assert status_resp.status == "blocked"
    assert status_resp.result is not None


def test_job_manager_ttl_cleanup():
    mgr = JobManager()
    mgr.create_job("job_expire")
    time.sleep(0.05)
    mgr.cleanup_expired(max_age_seconds=0.01)
    assert mgr.get_job("job_expire") is None
