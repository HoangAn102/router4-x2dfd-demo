import sys
import uuid
from pathlib import Path

# Ensure repository root is on sys.path regardless of CWD
repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import settings
from .routers import analyze_image, analyze_video, health, gpu_proxy
from .schemas.common import ErrorDetail, ErrorResponse
from .services.file_manager import cleanup_stale_temp_dirs, ensure_upload_dir
from .services.job_manager import job_manager
from .utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Lifespan
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} in RUN_MODE='{settings.RUN_MODE}'")
    logger.info("=" * 60)
    ensure_upload_dir()
    cleanup_stale_temp_dirs(max_age_hours=settings.TEMP_CLEANUP_MAX_AGE_HOURS)

    yield

    # Shutdown Lifespan
    logger.info("Shutting down API server; cleaning up active jobs...")
    job_manager.cleanup_expired(max_age_seconds=0)


app = FastAPI(
    title=settings.APP_NAME,
    description="Explainable Deepfake Detection API using Router4 and X²-DFD Multi-Expert Architecture",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS Middleware (supports local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global catch-all exception handler enforcing Fail-Closed JSON schema."""
    req_id = str(uuid.uuid4())
    logger.error(f"Unhandled Exception on {request.url.path} (request_id={req_id}): {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            request_id=req_id,
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                stage="global_handler",
                message="An unexpected server error occurred. Please contact the administrator with the request_id.",
                details=type(exc).__name__,
            )
        ).model_dump()
    )


# Register Routers
# Render same-origin proxy is intentionally a distinct /gpu-api path.
app.include_router(gpu_proxy.router)
app.include_router(health.router)
app.include_router(analyze_image.router)
app.include_router(analyze_video.router)

# Mount Frontend SPA if dist directory exists
from pathlib import Path
from fastapi.staticfiles import StaticFiles

frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

