import asyncio
import time
import uuid
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from ..config import settings
from ..schemas.common import ErrorDetail, ErrorResponse
from ..schemas.video import (
    VideoAnalyzeResponse,
    VideoInfo,
    VideoJobCreatedResponse,
    VideoJobStatusResponse,
)
from ..services.file_manager import ensure_upload_dir, isolated_temp_workspace
from ..services.job_manager import job_manager
from ..services.media_validator import MediaValidationError, MediaValidator
from ..utils.logger import logger
from web.model_worker.worker_factory import get_model_worker

from ..services.video_extractor import VideoExtractor
from ..services.video_aggregator import VideoAggregator
import shutil

router = APIRouter(prefix="/api/v1/analyze/video", tags=["Video Analysis (Async)"])


async def _run_video_background_task(job_id: str, video_path: Path):
    """Background task processing frames and aggregation for video."""
    t0 = time.time()
    frames_dir = video_path.parent / f"frames_{job_id}"
    try:
        worker = get_model_worker()

        # Stage 1: Extraction
        job_manager.update_progress(job_id, stage="extracting_frames", current_frame=0, total_frames=0)
        duration, frame_count, fps = MediaValidator.validate_video_file(video_path)

        extracted_data = VideoExtractor.extract_uniform_frames(video_path, frames_dir, target_num_frames=32)
        total_frames = len(extracted_data)
        frames = []

        # Stage 2:
        # LIVE worker uses batch-compute optimization while preserving
        # an independent Router/Expert/X2DFD result for every frame.
        if hasattr(worker, "analyze_video_batch"):

            def progress_callback(
                current,
                total,
                stage,
            ):
                job_manager.update_progress(
                    job_id,
                    stage=stage,
                    current_frame=current,
                    total_frames=total,
                )

            frames = await worker.analyze_video_batch(
                extracted_data,
                progress_callback=progress_callback,
            )

        else:

            # Compatibility path used by mock/tests only.
            for item in extracted_data:

                idx = item["frame_index"]
                timestamp = item["timestamp_seconds"]
                frame_path = item["frame_path"]
                thumb_b64 = item["thumbnail_b64"]

                job_manager.update_progress(
                    job_id,
                    stage="analyzing_frames",
                    current_frame=idx + 1,
                    total_frames=total_frames,
                )

                frame_res = (
                    await worker.analyze_video_frame(
                        frame_path,
                        idx,
                        timestamp,
                    )
                )

                frame_res.thumbnail_b64 = thumb_b64
                frames.append(frame_res)

                await asyncio.sleep(0.01)

        # Stage 3: Aggregation Gate
        job_manager.update_progress(job_id, stage="aggregating_results", current_frame=total_frames, total_frames=total_frames)
        final_result, error_code, error_msg = worker.aggregate_video(frames)

        elapsed_ms = (time.time() - t0) * 1000

        video_info = VideoInfo(
            duration_seconds=round(duration, 2),
            fps=round(fps, 1),
            total_frames_extracted=total_frames,
            frames_analyzed=total_frames,
            sampling_protocol="uniform_raw_frames_UNVERIFIED",
        )

        if final_result is not None:
            # Case A: Aggregation verified
            full_response = VideoAnalyzeResponse(
                request_id=job_id,
                status="ok",
                media_type="video",
                video=video_info,
                final=final_result,
                frames=frames,
                timing_ms={"total": round(elapsed_ms, 2)},
                provenance={"mode": settings.RUN_MODE},
            )
            job_manager.complete_job(job_id, full_response)
        else:
            # Case B: Aggregation UNVERIFIED (Fail-closed)
            blocked_response = VideoAnalyzeResponse(
                request_id=job_id,
                status="blocked",
                media_type="video",
                video=video_info,
                final=None,
                error_code=error_code or "VIDEO_AGGREGATION_UNVERIFIED",
                message=error_msg or "Video aggregation protocol is unverified from source.",
                frames=frames,
                timing_ms={"total": round(elapsed_ms, 2)},
                provenance={"mode": settings.RUN_MODE},
            )
            job_manager.block_job(job_id, blocked_response)

    except Exception as e:
        logger.error(f"Video job {job_id} encountered fatal error: {e}", exc_info=True)
        job_manager.fail_job(
            job_id,
            ErrorDetail(
                code="VIDEO_PROCESSING_FAILED",
                stage="video_pipeline",
                message=f"Video pipeline failed: {str(e)}",
                details=type(e).__name__,
            )
        )
    finally:
        # Clean up video file and extracted frames directory
        try:
            if frames_dir.exists():
                shutil.rmtree(frames_dir, ignore_errors=True)
            if video_path.exists():
                video_path.unlink()
        except Exception as e:
            logger.warning(f"Error during video background cleanup: {e}")


@router.post(
    "",
    response_model=VideoJobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {"model": ErrorResponse}}
)
async def analyze_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Initiate asynchronous video analysis job."""
    job_id = str(uuid.uuid4())
    logger.info(f"Received video upload request: {job_id} (filename={file.filename})")

    # Save to staging upload directory
    upload_root = ensure_upload_dir()
    staging_file = upload_root / f"job_{job_id}_{file.filename or 'upload.mp4'}"

    try:
        with open(staging_file, "wb") as buffer:
            while chunk := await file.read(2 * 1024 * 1024):  # 2MB chunks
                buffer.write(chunk)
    except Exception as e:
        logger.error(f"Failed to stream video {job_id}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                request_id=job_id,
                error=ErrorDetail(
                    code="FILE_SAVE_ERROR",
                    stage="upload_storage",
                    message="Failed to write uploaded video to storage.",
                )
            ).model_dump()
        )

    # Initial Validation
    try:
        MediaValidator.validate_video_file(staging_file)
    except MediaValidationError as mve:
        if staging_file.exists():
            staging_file.unlink()
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                request_id=job_id,
                error=ErrorDetail(
                    code=mve.code,
                    stage="media_validation",
                    message=mve.message,
                )
            ).model_dump()
        )

    # Create job & dispatch background worker
    job_manager.create_job(job_id)
    background_tasks.add_task(_run_video_background_task, job_id, staging_file)

    return VideoJobCreatedResponse(
        job_id=job_id,
        status="processing",
        poll_url=f"/api/v1/analyze/video/jobs/{job_id}",
        created_at=time.time(),
    )


@router.get(
    "/jobs/{job_id}",
    response_model=VideoJobStatusResponse,
    responses={404: {"model": ErrorResponse}}
)
async def get_video_job_status(job_id: str):
    """Poll video processing job status and results."""
    entry = job_manager.get_job(job_id)
    if not entry:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(
                request_id=job_id,
                error=ErrorDetail(
                    code="JOB_NOT_FOUND",
                    stage="job_lookup",
                    message=f"Video job '{job_id}' not found or expired.",
                )
            ).model_dump()
        )

    return entry.to_status_response()
