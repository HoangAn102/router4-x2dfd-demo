import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from web.backend.app.config import settings
from web.backend.app.schemas.common import ReadinessComponent
from web.backend.app.schemas.image import (
    ExpertResult,
    FinalResult,
    ImageAnalyzeResponse,
    RouterResult,
)
from web.backend.app.schemas.video import FrameResult, VideoFinalResult
from web.backend.app.services.video_aggregator import VideoAggregator
from web.backend.app.utils.gpu_lock import GpuLock
from web.backend.app.utils.logger import logger
from .base import BaseModelWorker
from .expert_client import ExpertClient
from .router_client import RouterClient
from .x2dfd_client import X2DFDClient


class LiveModelWorker(BaseModelWorker):
    """Production live model worker coordinating Router4, 4 Experts, Calibrators, and X²-DFD."""

    def __init__(self):
        self.router_client = RouterClient()
        self.expert_client = ExpertClient()
        self.x2dfd_client = X2DFDClient()
        self.aggregator = VideoAggregator()

    def check_readiness(self) -> Dict[str, ReadinessComponent]:
        """Check presence and readiness of physical artifacts on research server."""
        components = {}

        components["router_checkpoint"] = ReadinessComponent(
            name="Router4 Checkpoint (best.pt)",
            ready=Path(settings.ROUTER_CHECKPOINT).exists(),
            details=f"Path: {settings.ROUTER_CHECKPOINT}"
        )
        components["calibrators"] = ReadinessComponent(
            name="Teacher Calibrators (.joblib)",
            ready=Path(settings.CALIBRATORS_PATH).exists(),
            details=f"Path: {settings.CALIBRATORS_PATH}"
        )
        components["router4_lora"] = ReadinessComponent(
            name="Router4-aware LoRA Adapter",
            ready=Path(settings.ROUTER4_LORA_DIR).exists(),
            details=f"Path: {settings.ROUTER4_LORA_DIR}"
        )
        components["base_llava"] = ReadinessComponent(
            name="Base LLaVA-v1.5-7B Weights",
            ready=Path(settings.BASE_LLAVA_DIR).exists(),
            details=f"Path: {settings.BASE_LLAVA_DIR}"
        )
        components["x2python"] = ReadinessComponent(
            name="X²-DFD Python Runtime",
            ready=Path(settings.X2PYTHON_BIN).exists(),
            details=f"Path: {settings.X2PYTHON_BIN}"
        )

        return components

    async def analyze_image(self, image_path: Path, request_id: str) -> ImageAnalyzeResponse:
        t0 = time.time()
        logger.info(f"Starting live image analysis pipeline for {request_id} ({image_path.name})")

        async with GpuLock():
            # Step 1: Router4 classifies image into 1 of 4 experts
            t_route_0 = time.time()
            selected_expert, selected_alias, router_conf, margin = self.router_client.route_image(image_path)
            t_route = (time.time() - t_route_0) * 1000

            # Step 2: Selected Expert runs and Calibrator scales raw score
            t_exp_0 = time.time()
            raw_score, calibrated_score = await self.expert_client.compute_expert_score(selected_expert, image_path)
            t_exp = (time.time() - t_exp_0) * 1000

            # Step 3: WFS prompt constructed & X²-DFD LLaVA + LoRA infers continuous score
            t_mllm_0 = time.time()
            final_result = await self.x2dfd_client.infer(image_path, selected_alias, calibrated_score)
            t_mllm = (time.time() - t_mllm_0) * 1000

        total_ms = (time.time() - t0) * 1000

        return ImageAnalyzeResponse(
            request_id=request_id,
            status="ok",
            media_type="image",
            router=RouterResult(
                selected_expert=selected_expert,
                selected_alias=selected_alias,  # type: ignore
                router_confidence=router_conf,
                margin=margin,
            ),
            expert=ExpertResult(
                name=selected_expert,
                raw_score=raw_score,
                calibrated_score=calibrated_score,
                score_direction="fake_probability",
            ),
            final=final_result,
            timing_ms={
                "routing": round(t_route, 2),
                "expert_inference": round(t_exp, 2),
                "x2dfd_inference": round(t_mllm, 2),
                "total": round(total_ms, 2),
            },
            provenance={
                "mode": "live_server",
                "router_checkpoint": Path(settings.ROUTER_CHECKPOINT).name,
                "lora_adapter": Path(settings.ROUTER4_LORA_DIR).name,
            }
        )

    async def analyze_video_frame(
        self, frame_path: Path, frame_idx: int, timestamp: float
    ) -> FrameResult:
        """Run single-frame through the complete Router -> Expert -> Calibrator -> X²-DFD pipeline."""
        async with GpuLock():
            selected_expert, selected_alias, _, _ = self.router_client.route_image(frame_path)
            raw_score, calibrated_score = await self.expert_client.compute_expert_score(selected_expert, frame_path)
            final_result = await self.x2dfd_client.infer(frame_path, selected_alias, calibrated_score)

        return FrameResult(
            frame_index=frame_idx,
            timestamp_seconds=timestamp,
            selected_expert=selected_expert,
            expert_score=calibrated_score,
            final_frame_score=final_result.continuous_score,
            verdict=final_result.verdict,
            thumbnail_b64=None,
        )

    def aggregate_video(
        self, frames: List[FrameResult]
    ) -> Tuple[Optional[VideoFinalResult], Optional[str], Optional[str]]:
        """Delegate video-level aggregation to the verified aggregator service."""
        return self.aggregator.aggregate(frames)
