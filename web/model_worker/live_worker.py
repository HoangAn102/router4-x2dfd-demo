import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
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

    def _should_release_x2dfd_for_experts(self) -> bool:
        """
        Keep X2DFD resident on large-memory GPUs.

        A40 48GB has enough headroom for the persistent LLaVA/LoRA
        plus the project's selected forensic experts in the current
        single-request design.

        Smaller GPUs retain the conservative release behavior.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return True

            total_gb = (
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3)
            )

            release = total_gb < 40.0

            logger.info(
                "GPU memory %.1f GB -> X2DFD %s before expert stage",
                total_gb,
                "RELEASE" if release else "KEEP-WARM",
            )

            return release

        except Exception:
            return True


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

            # Step 2: selected expert.
            #
            # Release persistent X2DFD before loading a forensic expert
            # to avoid unnecessary simultaneous VRAM residency.
            if self._should_release_x2dfd_for_experts():
                await self.x2dfd_client.stop_worker()
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


    async def analyze_video_batch(
        self,
        extracted_data: List[dict],
        progress_callback: Optional[
            Callable[[int, int, str], None]
        ] = None,
    ) -> List[FrameResult]:

        """
        Optimized but scientifically equivalent per-frame video path.

        Every frame still performs:
          Router4 -> selected expert -> calibration -> X2DFD.

        Optimization only:
          - Router4 stays loaded.
          - Frames selecting the same expert are batched.
          - X2DFD/LLaVA loads once for the X2DFD frame stage.
        """

        if not extracted_data:
            return []

        async with GpuLock():

            total = len(extracted_data)

            routed = []

            groups = {
                "blending": [],
                "diffusion": [],
                "frequency": [],
                "texture": [],
            }


            # ----------------------------------------------
            # A. Independent Router4 decision per frame.
            # ----------------------------------------------

            for pos, item in enumerate(
                extracted_data
            ):

                path = (
                    Path(item["frame_path"])
                    .expanduser()
                    .resolve(strict=True)
                )

                (
                    expert,
                    alias,
                    router_conf,
                    router_margin,
                ) = self.router_client.route_image(
                    path
                )

                routed.append(
                    {
                        "item": item,
                        "path": path,
                        "expert": expert,
                        "alias": alias,
                        "router_confidence":
                            router_conf,
                        "router_margin":
                            router_margin,
                    }
                )

                groups[expert].append(
                    path
                )

                if progress_callback:
                    progress_callback(
                        pos + 1,
                        total,
                        "routing_frames",
                    )


            # Ensure forensic expert batches get maximum free VRAM.
            if self._should_release_x2dfd_for_experts():
                await self.x2dfd_client.stop_worker()
            # ----------------------------------------------
            # B. Batch only computationally.
            #    Scores remain one-per-frame.
            # ----------------------------------------------

            scores = {}

            active_groups = [
                (expert, groups[expert])
                for expert in (
                    "blending",
                    "diffusion",
                    "frequency",
                    "texture",
                )
                if groups[expert]
            ]

            for expert_pos, (expert, paths) in enumerate(
                active_groups,
                start=1,
            ):

                if progress_callback:
                    progress_callback(
                        expert_pos,
                        len(active_groups),
                        "scoring_experts",
                    )

                batch = (
                    await self.expert_client
                    .compute_expert_scores_batch(
                        expert,
                        paths,
                    )
                )

                scores.update(batch)


            # ----------------------------------------------
            # C. Independent X2DFD decision per frame.
            #    First frame loads LLaVA+LoRA.
            #    Remaining frames reuse cached model.
            # ----------------------------------------------

            frames = []

            if progress_callback:
                progress_callback(
                    0,
                    total,
                    "loading_x2dfd",
                )

            for pos, row in enumerate(
                routed
            ):

                key = str(
                    row["path"]
                )

                if key not in scores:
                    raise RuntimeError(
                        f"Missing expert score: {key}"
                    )

                _, calibrated = scores[key]

                final_result = (
                    await self.x2dfd_client.infer(
                        row["path"],
                        row["alias"],
                        calibrated,
                        include_explanation=False,
                    )
                )

                item = row["item"]

                frames.append(
                    FrameResult(
                        frame_index=
                            item["frame_index"],

                        timestamp_seconds=
                            item["timestamp_seconds"],

                        selected_expert=
                            row["expert"],

                        expert_score=
                            calibrated,

                        final_frame_score=
                            final_result.continuous_score,

                        verdict=
                            final_result.verdict,

                        thumbnail_b64=
                            item.get("thumbnail_b64"),
                    )
                )

                if progress_callback:
                    progress_callback(
                        pos + 1,
                        total,
                        "analyzing_frames",
                    )

            return frames


    async def analyze_video_frame(
        self, frame_path: Path, frame_idx: int, timestamp: float
    ) -> FrameResult:
        """Run single-frame through the complete Router -> Expert -> Calibrator -> X²-DFD pipeline."""
        async with GpuLock():
            selected_expert, selected_alias, _, _ = self.router_client.route_image(frame_path)
            raw_score, calibrated_score = await self.expert_client.compute_expert_score(selected_expert, frame_path)
            final_result = await self.x2dfd_client.infer(
                frame_path,
                selected_alias,
                calibrated_score,
                include_explanation=False,
            )

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
