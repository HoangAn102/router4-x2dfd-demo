import uuid
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from ..schemas.common import ErrorDetail, ErrorResponse
from ..schemas.image import ImageAnalyzeResponse
from ..services.file_manager import isolated_temp_workspace
from ..services.media_validator import MediaValidationError, MediaValidator
from ..utils.logger import logger
from web.model_worker.worker_factory import get_model_worker

router = APIRouter(prefix="/api/v1/analyze", tags=["Image Analysis"])


@router.post(
    "/image",
    response_model=ImageAnalyzeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """Analyze a single uploaded image through Router4 and the X²-DFD pipeline."""
    request_id = str(uuid.uuid4())
    logger.info(f"Received image analysis request: {request_id} (filename={file.filename})")

    async with isolated_temp_workspace(prefix=f"img_{request_id[:8]}") as workspace:
        raw_name = file.filename or "upload.jpg"
        safe_name = Path(raw_name).name

        if safe_name in {"", ".", ".."}:
            safe_name = "upload.jpg"

        file_path = workspace / safe_name

        # Stream save file
        try:
            with open(file_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    buffer.write(chunk)
        except Exception as e:
            logger.error(f"Error saving uploaded file {request_id}: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    request_id=request_id,
                    error=ErrorDetail(
                        code="FILE_SAVE_ERROR",
                        stage="upload_storage",
                        message="Failed to write uploaded file to temporary workspace.",
                        details=str(e),
                    )
                ).model_dump()
            )

        # Validation stage
        try:
            final_width, final_height = MediaValidator.prepare_image_file(
                file_path
            )

            logger.info(
                "Prepared image %s for inference: %dx%d",
                request_id,
                final_width,
                final_height,
            )
        except MediaValidationError as mve:
            logger.warning(f"Media validation failed for {request_id}: {mve.code} - {mve.message}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    request_id=request_id,
                    error=ErrorDetail(
                        code=mve.code,
                        stage="media_validation",
                        message=mve.message,
                    )
                ).model_dump()
            )

        # Model Inference stage
        try:
            worker = get_model_worker()
            result = await worker.analyze_image(file_path, request_id)
            return result
        except Exception as e:
            logger.error(f"Inference pipeline failed for {request_id}: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    request_id=request_id,
                    error=ErrorDetail(
                        code="INFERENCE_FAILED",
                        stage="model_pipeline",
                        message=f"Deepfake detection pipeline failed: {str(e)}",
                        details=type(e).__name__,
                    )
                ).model_dump()
            )
