from fastapi import APIRouter, Response, status
from ..config import settings
from ..schemas.common import HealthResponse, ReadyResponse
from web.model_worker.worker_factory import get_model_worker

router = APIRouter(prefix="/api/v1", tags=["Health & Readiness"])


@router.get("/health", response_model=HealthResponse)
async def check_health():
    """Verify that the API server process is alive."""
    return HealthResponse(
        status="alive",
        app_name=settings.APP_NAME,
        run_mode=settings.RUN_MODE,
        version="3.0.0",
    )


@router.get("/ready", response_model=ReadyResponse)
async def check_ready(response: Response):
    """Verify operational readiness of model worker and execution environments."""
    try:
        worker = get_model_worker()
        components = worker.check_readiness()
        all_ready = all(comp.ready for comp in components.values())
    except Exception as e:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            status="not_ready",
            run_mode=settings.RUN_MODE,
            all_ready=False,
            components={
                "worker_init": {
                    "name": "Worker Initialization",
                    "ready": False,
                    "details": str(e)
                }  # type: ignore
            }
        )

    if not all_ready and settings.RUN_MODE == "live":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        status_str = "not_ready"
    else:
        status_str = "ready"

    return ReadyResponse(
        status=status_str,
        run_mode=settings.RUN_MODE,
        all_ready=all_ready,
        components=components,
    )
