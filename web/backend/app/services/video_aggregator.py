from typing import List, Optional, Tuple
from ..schemas.video import FrameResult, VideoFinalResult
from ..utils.logger import logger


class VideoAggregator:
    """
    Video Aggregation Engine.
    Enforces the fail-closed invariant: If no aggregation formula has been
    empirically verified from the research codebase, returns BLOCKED status
    and refuses to fabricate a video-level verdict or continuous score.
    """

    def __init__(self, verified_protocol: Optional[str] = None):
        # Default is None since Sprint 0 confirmed aggregation is UNVERIFIED in the snapshot
        self.verified_protocol = verified_protocol

    def aggregate(
        self, frames: List[FrameResult]
    ) -> Tuple[Optional[VideoFinalResult], Optional[str], Optional[str]]:
        """
        Aggregate per-frame results into a video-level result.
        Returns: (VideoFinalResult | None, error_code | None, error_message | None)
        """
        if not frames:
            return (
                None,
                "NO_FRAMES_ANALYZED",
                "No valid frames were available for video analysis."
            )

        # Invariant Check: Fail-closed if aggregation protocol is not verified
        if self.verified_protocol is None:
            logger.warning(
                "Video aggregation requested, but no verified aggregation algorithm exists in source. "
                "Returning BLOCKED status."
            )
            return (
                None,
                "VIDEO_AGGREGATION_UNVERIFIED",
                "Video aggregation algorithm is unverified from repository research source. "
                "Frame-level forensic results are preserved."
            )

        # Future extensible branch if a verified protocol is provided:
        if self.verified_protocol == "mean_probability":
            avg_fake = sum(f.final_frame_score for f in frames) / len(frames)
            avg_real = 1.0 - avg_fake
            verdict = "FAKE" if avg_fake >= 0.5 else "REAL"
            return (
                VideoFinalResult(
                    verdict=verdict,
                    fake_probability=round(avg_fake, 6),
                    real_probability=round(avg_real, 6),
                    continuous_score=round(avg_fake, 6),
                    aggregation_method="verified_mean_probability",
                ),
                None,
                None,
            )

        return (
            None,
            "UNKNOWN_AGGREGATION_PROTOCOL",
            f"Unsupported video aggregation protocol '{self.verified_protocol}'."
        )
