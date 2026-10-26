from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from web.backend.app.schemas.common import ReadinessComponent
from web.backend.app.schemas.image import ImageAnalyzeResponse
from web.backend.app.schemas.video import FrameResult, VideoFinalResult


class BaseModelWorker(ABC):
    """Abstract interface for all model workers (Mock & Live)."""

    @abstractmethod
    def check_readiness(self) -> Dict[str, ReadinessComponent]:
        """Verify readiness of all necessary artifacts and environments."""
        pass

    @abstractmethod
    async def analyze_image(self, image_path: Path, request_id: str) -> ImageAnalyzeResponse:
        """Run the end-to-end image inference pipeline."""
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
