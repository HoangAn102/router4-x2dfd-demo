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

    # Face-crop preprocessing (models were trained on DeepfakeBench face crops)
    FACE_CROP_BACKEND: Literal["dlib", "haar", "auto"] = "auto"
    DEEPFAKEBENCH_PREPROCESS_DIR: str = "/home/aiotlab/hoangan/projects/DeepfakeBench/preprocessing"
    DLIB_PREDICTOR_PATH: str = ""
    FACE_CROP_ENABLED: bool = True
    FACE_CROP_SCALE: float = 1.3
    FACE_CROP_SIZE: int = 256
    FACE_MIN_SIZE: int = 60
    # Fallback when strict Haar finds no face (OpenCV YuNet, MIT, opencv_zoo 2023mar).
    # 2026-10-26: found the face on 2/3 phone selfies Haar missed, 0/39 hits on face-free images.
    FACE_YUNET_MODEL: str = "/home/aiotlab/hoangan/models/yunet/face_detection_yunet_2023mar.onnx"
    FACE_YUNET_SCORE: float = 0.9
    # Haar boxes smaller than this fraction of the short side are re-checked with YuNet.
    FACE_HAAR_MIN_FRACTION: float = 0.10

    # Reliability flag (presentation only; does NOT change routing, scores or verdict).
    # 0.30 = bottom ~18% router confidence on FFPP val (AUC 0.67 there vs 0.92 overall).
    # Previously 0.38, the threshold the team used in
    # source_snapshot/final_clean_integration/03_wfs/patched_route_test_locked038.py
    ROUTER_LOW_CONFIDENCE: float = 0.30
    # Disabled: calibrated floors (blending 0.45, texture 0.56) sit inside 0.4-0.6,
    # so this flagged 51% of REAL val images (152/300). Keep 0.0 unless re-validated.
    EXPERT_NEUTRAL_BAND: float = 0.0

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
    # The calibrator that produced the scores in the LoRA's SFT prompts. The FINAL_CLEAN SFT json is a byte copy
    # of ROUTER4_NATIVE_X2DFD_24K_20260915 (sha 05814535…), whose scores come from router_teacher_v2
    # (median |diff| <= 0.002 vs 0.011-0.047 for calibrators_FINAL_CLEAN.joblib; audit 2026-10-30, AUDIT.md P3).
    CALIBRATORS_PATH: str = "/home/aiotlab/hoangan/outputs/router_teacher_v2/calibrators_train_only.joblib"
    ROUTER4_LORA_DIR: str = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/05_lora/router4_x2dfd_FINAL_CLEAN"
    BASE_LLAVA_DIR: str = "/home/aiotlab/hoangan/projects/X2DFD/weights/base/llava-v1.5-7b"

    # Python Execution Wrappers (Verified from runtime_clean_v2)
    X2PYTHON_BIN: str = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/runtime_clean_v2/x2python"
    FREQPYTHON_BIN: str = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/runtime_clean_v2/freqpython"
    TEXPYTHON_BIN: str = "/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/runtime_clean_v2/texpython"

    # LIVE research source roots
    X2DFD_PROJECT_ROOT: str = "/home/aiotlab/hoangan/projects/X2DFD"
    DFFREQ_PROJECT_ROOT: str = "/home/aiotlab/hoangan/projects/DFFreq-main"
    TEXTURE_PROJECT_ROOT: str = "/home/aiotlab/hoangan/projects/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild/stylegan-ffhq"
    ROUTER8_WRAPPER_ROOT: str = "/home/aiotlab/hoangan/router8_x2crop"

    # Public GPU endpoint used ONLY by the Render same-origin proxy.
    GPU_API_ORIGIN: str = 'https://gains-how-cheque-enquiry.trycloudflare.com'

    # First load of large expert/MLLM checkpoints can be slow.
    EXPERT_INFERENCE_TIMEOUT_SEC: int = 600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
