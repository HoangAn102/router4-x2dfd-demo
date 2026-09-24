import time
from typing import Dict, Optional
from ..schemas.common import ErrorDetail
from ..schemas.video import VideoAnalyzeResponse, VideoJobStatusResponse
from ..utils.logger import logger


class JobEntry:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status: str = "processing"
        self.stage: str = "queued"
        self.current_frame: int = 0
        self.total_frames: int = 0
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.result: Optional[VideoAnalyzeResponse] = None
        self.error: Optional[ErrorDetail] = None

    def to_status_response(self) -> VideoJobStatusResponse:
        return VideoJobStatusResponse(
            job_id=self.job_id,
            status=self.status,  # type: ignore
            stage=self.stage,
            current_frame=self.current_frame,
            total_frames=self.total_frames,
            result=self.result,
            error=self.error,
        )


class JobManager:
    """Thread-safe manager for asynchronous video processing jobs."""

    def __init__(self):
        self._jobs: Dict[str, JobEntry] = {}

    def create_job(self, job_id: str) -> JobEntry:
        entry = JobEntry(job_id)
        self._jobs[job_id] = entry
        logger.info(f"Video job created: {job_id}")
        return entry

    def get_job(self, job_id: str) -> Optional[JobEntry]:
        return self._jobs.get(job_id)

    def update_progress(self, job_id: str, stage: str, current_frame: int = 0, total_frames: int = 0):
        entry = self.get_job(job_id)
        if entry:
            entry.stage = stage
            entry.current_frame = current_frame
            entry.total_frames = total_frames
            entry.updated_at = time.time()

    def complete_job(self, job_id: str, result: VideoAnalyzeResponse):
        entry = self.get_job(job_id)
        if entry:
            entry.status = "completed"
            entry.stage = "completed"
            entry.result = result
            entry.updated_at = time.time()
            logger.info(f"Video job completed successfully: {job_id}")

    def block_job(self, job_id: str, result: VideoAnalyzeResponse):
        entry = self.get_job(job_id)
        if entry:
            entry.status = "blocked"
            entry.stage = "blocked_aggregation_unverified"
            entry.result = result
            entry.updated_at = time.time()
            logger.warning(f"Video job completed with BLOCKED status: {job_id}")

    def fail_job(self, job_id: str, error: ErrorDetail):
        entry = self.get_job(job_id)
        if entry:
            entry.status = "failed"
            entry.stage = "failed"
            entry.error = error
            entry.updated_at = time.time()
            logger.error(f"Video job failed: {job_id} - {error.code}: {error.message}")

    def cleanup_expired(self, max_age_seconds: int = 900):
        now = time.time()
        to_delete = [
            jid for jid, entry in self._jobs.items()
            if (now - entry.created_at) > max_age_seconds
        ]
        for jid in to_delete:
            del self._jobs[jid]
        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} expired video jobs.")


job_manager = JobManager()
