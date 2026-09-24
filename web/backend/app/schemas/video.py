import math
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from .common import ErrorDetail


class FrameResult(BaseModel):
    frame_index: int
    timestamp_seconds: float
    selected_expert: Literal["blending", "diffusion", "frequency", "texture"]
    expert_score: float = Field(..., ge=0.0, le=1.0)
    final_frame_score: float = Field(..., ge=0.0, le=1.0)
    verdict: Literal["REAL", "FAKE"]
    thumbnail_b64: Optional[str] = Field(None, description="Optional 128x128 JPEG Base64 thumbnail")

    @field_validator("expert_score", "final_frame_score", mode="before")
    @classmethod
    def validate_finite_and_non_bool(cls, v):
        if isinstance(v, bool):
            raise ValueError("Score cannot be a boolean value")
        if not isinstance(v, (int, float)):
            raise ValueError(f"Score must be numeric, got {type(v).__name__}")
        f_val = float(v)
        if not math.isfinite(f_val):
            raise ValueError(f"Score must be finite, got {f_val}")
        if not (0.0 <= f_val <= 1.0):
            raise ValueError(f"Score must be in range [0, 1], got {f_val}")
        return f_val


class VideoFinalResult(BaseModel):
    verdict: Literal["REAL", "FAKE"]
    fake_probability: float = Field(..., ge=0.0, le=1.0)
    real_probability: float = Field(..., ge=0.0, le=1.0)
    continuous_score: float = Field(..., ge=0.0, le=1.0)
    aggregation_method: str
    decision_threshold: float = 0.5
    explanation: Optional[str] = None

    @field_validator("fake_probability", "real_probability", "continuous_score", mode="before")
    @classmethod
    def validate_finite_and_non_bool(cls, v):
        if isinstance(v, bool):
            raise ValueError("Score cannot be a boolean value")
        if not isinstance(v, (int, float)):
            raise ValueError(f"Score must be numeric, got {type(v).__name__}")
        f_val = float(v)
        if not math.isfinite(f_val):
            raise ValueError(f"Score must be finite, got {f_val}")
        if not (0.0 <= f_val <= 1.0):
            raise ValueError(f"Score must be in range [0, 1], got {f_val}")
        return f_val


class VideoInfo(BaseModel):
    duration_seconds: float
    fps: float
    total_frames_extracted: int
    frames_analyzed: int
    sampling_protocol: str = "DeepfakeBench_fixed32_uniform"


class VideoAnalyzeResponse(BaseModel):
    request_id: str
    status: Literal["ok", "blocked"]
    media_type: Literal["video"] = "video"
    video: VideoInfo
    final: Optional[VideoFinalResult] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    frames: List[FrameResult]
    timing_ms: Dict[str, float]
    provenance: Optional[Dict[str, str]] = None


class VideoJobCreatedResponse(BaseModel):
    job_id: str
    status: Literal["processing"] = "processing"
    poll_url: str
    created_at: float


class VideoJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["processing", "completed", "blocked", "failed"]
    stage: Optional[str] = None
    current_frame: Optional[int] = None
    total_frames: Optional[int] = None
    result: Optional[VideoAnalyzeResponse] = None
    error: Optional[ErrorDetail] = None
