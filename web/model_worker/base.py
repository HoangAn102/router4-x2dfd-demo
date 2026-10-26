import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from web.backend.app.schemas.common import ReadinessComponent
from web.backend.app.schemas.image import ImageAnalyzeResponse
from web.backend.app.schemas.video import FrameResult, VideoFinalResult
from web.backend.app.services.ai_gate import build_faceless_response


class BaseModelWorker(ABC):
    """Abstract interface for all model workers (Mock & Live)."""

    PROVENANCE_MODE = "unknown"

    async def score_full_image(self, full_image_path: Path) -> Optional[float]:
        """AI-image probability of the whole upload, or None when the gate is off/unavailable."""
        return None

    async def analyze_faceless_image(
        self, full_image_path: Path, request_id: str
    ) -> Optional[ImageAnalyzeResponse]:
        """No face found: FAKE response if the AI-image gate fires, else None (keep NO_FACE_DETECTED)."""
        t0 = time.time()
        ai_prob = await self.score_full_image(full_image_path)
        elapsed = round((time.time() - t0) * 1000, 2)
        return build_faceless_response(
            request_id,
            ai_prob,
            timing_ms={"ai_detector": elapsed, "total": elapsed},
            provenance={"mode": self.PROVENANCE_MODE, "pipeline": "ai_gate_only"},
        )

    @abstractmethod
    def check_readiness(self) -> Dict[str, ReadinessComponent]:
        """Verify readiness of all necessary artifacts and environments."""
        pass

    @abstractmethod
    async def analyze_image(
        self, image_path: Path, request_id: str, full_image_path: Optional[Path] = None
    ) -> ImageAnalyzeResponse:
        """Run the end-to-end image inference pipeline.

        image_path is the face crop fed to Router4/experts/X2DFD; full_image_path is the
        whole upload (before cropping) used by the general AI-image gate.
        """
        pass

    @abstractmethod
    async def analyze_video_frame(
        self, frame_path: Path, frame_idx: int, timestamp: float
    ) -> FrameResult:
        """Run the single-frame inference pipeline."""
        pass

    @abstractmethod
    def aggregate_video(
        self, frames: List[FrameResult]
    ) -> Tuple[Optional[VideoFinalResult], Optional[str], Optional[str]]:
        """
        Aggregate per-frame results into a video-level verdict.
        Returns: (final_result, error_code, error_message)
        If unverified, final_result must be None with appropriate error_code.
        """
        pass
