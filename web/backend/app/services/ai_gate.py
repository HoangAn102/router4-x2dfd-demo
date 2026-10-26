"""
General AI-image gate applied after the X2DFD verdict.

X2DFD/Router4 were trained on FF++/GAN/StyleGAN only, so diffusion images
(Midjourney, SD...) are scored REAL. A general AI-image detector runs on the
full upload; a confident hit forces the verdict to FAKE, a borderline hit only
lowers reliability. With AI_GATE_ENABLED=false the detector is not run and the
response is exactly the X2DFD one.
"""

from pathlib import Path
from typing import Optional, Tuple

from ..config import settings
from ..schemas.image import AIDetectorResult, FinalResult, Reliability


def apply_ai_gate(
    final: FinalResult,
    reliability: Reliability,
    ai_prob: Optional[float],
) -> Tuple[FinalResult, Reliability, Optional[AIDetectorResult]]:
    if ai_prob is None:
        return final, reliability, None

    threshold = float(settings.AI_GATE_THRESHOLD)
    warn = float(settings.AI_GATE_WARN)
    detector = AIDetectorResult(
        score=ai_prob,
        threshold=threshold,
        model=Path(settings.AI_GATE_MODEL).name,
    )

    if ai_prob >= threshold:
        detector.overridden = True
        detector.x2dfd_fake_probability = final.fake_probability
        final = final.model_copy(update={
            "verdict": "FAKE",
            "fake_probability": ai_prob,
            "real_probability": 1.0 - ai_prob,
            "continuous_score": ai_prob,
            "explanation": (
                "Phát hiện dấu hiệu ảnh do AI tạo (detector tổng quát), "
                f"điểm {ai_prob:.3f}."
            ),
        })
    elif ai_prob >= warn:
        reliability = Reliability(
            level="low",
            reasons=reliability.reasons + [
                f"Có dấu hiệu ảnh do AI tạo (điểm {ai_prob:.3f}), chưa đủ chắc chắn."
            ],
        )

    return final, reliability, detector
