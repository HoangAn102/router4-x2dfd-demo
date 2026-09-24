import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "Router4-X2DFD-Deepfake-Detection"
    APP_ENV: str = "development"
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Core Execution Mode (Single standardized variable: "mock" or "live")
    RUN_MODE: Literal["mock", "live"] = "mock"

    # Media Limits (Shared between FE and BE)
    MAX_IMAGE_SIZE_MB: int = 20
    MAX_VIDEO_SIZE_MB: int = 100
    MAX_VIDEO_DURATION_SEC: int = 60
    MAX_IMAGE_DIMENSION: int = 4096

    # Multi-Tier Timeouts (seconds)
    IMAGE_INFERENCE_TIMEOUT_SEC: int = 60
    FRAME_INFERENCE_TIMEOUT_SEC: int = 45
    VIDEO_TOTAL_TIMEOUT_SEC: int = 300

    # Video Job & Temp Settings
    JOB_TTL_SECONDS: int = 900  # 15 minutes
    UPLOAD_DIR: Path = Path("uploads")
    TEMP_CLEANUP_MAX_AGE_HOURS: int = 1

    # GPU & Subprocess Concurrency
    MAX_CONCURRENT_GPU_TASKS: int = 1
    GPU_LOCK_FILE: Path = Path("gpu.lock")

    # Artifact & Checkpoint Paths (Verified from runtime_context & source_snapshot)
    ROUTER_CHECKPOINT: str = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/02_router/final_checkpoint/best.pt"
    CALIBRATORS_PATH: str = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/01_teacher/calibrators_FINAL_CLEAN.joblib"
    ROUTER4_LORA_DIR: str = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/05_lora/router4_x2dfd_FINAL_CLEAN"
    BASE_LLAVA_DIR: str = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"

    # Python Execution Wrappers (Verified from runtime_clean_v2)
    X2PYTHON_BIN: str = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/runtime_clean_v2/x2python"
    FREQPYTHON_BIN: str = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/runtime_clean_v2/freqpython"
    TEXPYTHON_BIN: str = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/runtime_clean_v2/texpython"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
