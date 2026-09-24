from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Unique machine-readable error code")
    stage: str = Field(..., description="Pipeline stage where the error occurred")
    message: str = Field(..., description="Human-readable explanation of the error")
    details: Optional[str] = Field(None, description="Technical diagnostics or traceback snippet")


class ErrorResponse(BaseModel):
    request_id: str
    status: str = "error"
    error: ErrorDetail
    timing_ms: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    status: str = "alive"
    app_name: str
    run_mode: str
    version: str = "3.0.0"


class ReadinessComponent(BaseModel):
    name: str
    ready: bool
    details: Optional[str] = None


class ReadyResponse(BaseModel):
    status: str  # "ready" or "not_ready"
    run_mode: str
    all_ready: bool
    components: Dict[str, ReadinessComponent]
