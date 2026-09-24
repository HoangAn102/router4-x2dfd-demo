import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from web.backend.app.schemas.common import ReadinessComponent
from web.backend.app.schemas.image import (
    ExpertResult,
    FinalResult,
    ImageAnalyzeResponse,
    RouterResult,
)
from web.backend.app.schemas.video import FrameResult, VideoFinalResult
from .base import BaseModelWorker


class MockModelWorker(BaseModelWorker):
    """Deterministic Mock Worker strictly for local development and unit tests."""

    EXPERTS = ["blending", "diffusion", "frequency", "texture"]
    ALIASES = {
        "blending": "Blending",
        "diffusion": "Diffusion",
        "frequency": "Frequency",
        "texture": "Texture",
    }

    def check_readiness(self) -> Dict[str, ReadinessComponent]:
        return {
            "mock_engine": ReadinessComponent(
                name="Mock Engine",
                ready=True,
                details="Mock worker active for local development/testing."
            )
        }

    def _hash_to_unit_float(self, data: bytes) -> float:
        h = int(hashlib.sha256(data).hexdigest()[:8], 16)
        return round((h % 10000) / 10000.0, 4)

    async def analyze_image(self, image_path: Path, request_id: str) -> ImageAnalyzeResponse:
        t0 = time.time()
        file_bytes = image_path.read_bytes() if image_path.exists() else b"mock_dummy"
        val = self._hash_to_unit_float(file_bytes)

        # Deterministic expert assignment
        expert_idx = int(val * 4) % 4
        selected_expert = self.EXPERTS[expert_idx]
        selected_alias = self.ALIASES[selected_expert]

        router_conf = round(0.70 + 0.28 * val, 4)
        margin = round(router_conf - 0.25, 4)
        raw_score = round(val, 4)
        calibrated_score = round(min(1.0, max(0.0, val * 0.95 + 0.02)), 4)

        # Final continuous score
        fake_prob = round(val, 6)
        real_prob = round(1.0 - fake_prob, 6)
        verdict = "FAKE" if fake_prob >= 0.5 else "REAL"

        # Model explanation: only set if hash condition met, else None
        explanation = (
            f"Forensic cues indicate manipulation consistent with {selected_alias} artifacts."
            if val > 0.8 else None
        )

        elapsed_ms = (time.time() - t0) * 1000 + 45.0  # slight simulated latency

        return ImageAnalyzeResponse(
            request_id=request_id,
            status="ok",
            media_type="image",
            router=RouterResult(
                selected_expert=selected_expert,  # type: ignore
                selected_alias=selected_alias,  # type: ignore
                router_confidence=router_conf,
                margin=margin,
            ),
            expert=ExpertResult(
                name=selected_expert,  # type: ignore
                raw_score=raw_score,
                calibrated_score=calibrated_score,
                score_direction="fake_probability",
            ),
            final=FinalResult(
                verdict=verdict,
                fake_probability=fake_prob,
                real_probability=real_prob,
                continuous_score=fake_prob,
                decision_threshold=0.5,
                explanation=explanation,
            ),
            timing_ms={
                "validation": 5.0,
                "routing": 15.0,
                "expert_inference": 20.0,
                "x2dfd_inference": 30.0,
                "total": round(elapsed_ms, 2),
            },
            provenance={
                "mode": "mock_development",
                "notice": "Simulated output for UI/API integration testing only."
            }
        )

    async def analyze_video_frame(
        self, frame_path: Path, frame_idx: int, timestamp: float
    ) -> FrameResult:
        file_bytes = frame_path.read_bytes() if frame_path.exists() else f"frame_{frame_idx}".encode()
        val = self._hash_to_unit_float(file_bytes)

        expert_idx = int((val + frame_idx * 0.1) * 4) % 4
        selected_expert = self.EXPERTS[expert_idx]
        fake_prob = round(val, 4)
        verdict = "FAKE" if fake_prob >= 0.5 else "REAL"

        return FrameResult(
            frame_index=frame_idx,
            timestamp_seconds=timestamp,
            selected_expert=selected_expert,  # type: ignore
            expert_score=fake_prob,
            final_frame_score=fake_prob,
            verdict=verdict,
            thumbnail_b64=None,
        )

    def aggregate_video(
        self, frames: List[FrameResult]
    ) -> Tuple[Optional[VideoFinalResult], Optional[str], Optional[str]]:
        # In alignment with the audit findings: Aggregation is UNVERIFIED in the snapshot
        # Return None to trigger status="blocked" and fail-closed behavior
        return (
            None,
            "VIDEO_AGGREGATION_UNVERIFIED",
            "Video aggregation algorithm has not been verified from repository research source."
        )
