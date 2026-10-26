"""
General AI-image gate applied after the X2DFD verdict.

X2DFD/Router4 were trained on FF++/GAN/StyleGAN only, so diffusion images
(Midjourney, SD...) are scored REAL. A general AI-image detector runs on the
full upload; a confident hit forces the verdict to FAKE, a borderline hit only
lowers reliability. With AI_GATE_ENABLED=false the detector is not run and the
response is exactly the X2DFD one.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

from ..config import settings
from ..schemas.image import AIDetectorResult, FinalResult, ImageAnalyzeResponse, Reliability


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


def build_faceless_response(
    request_id: str,
    ai_prob: Optional[float],
    timing_ms: Dict[str, float],
    provenance: Dict[str, str],
) -> Optional[ImageAnalyzeResponse]:
    """
    Result for an upload with no detectable face.

    Router4/experts/X2DFD need a face crop, so only the AI-image gate can judge.
    Returns a FAKE response when ai_prob >= AI_GATE_THRESHOLD; otherwise None,
    and the caller keeps the NO_FACE_DETECTED error.
    """
    if ai_prob is None or ai_prob < float(settings.AI_GATE_THRESHOLD):
        return None

    final = FinalResult(
        verdict="FAKE",
        fake_probability=ai_prob,
        real_probability=1.0 - ai_prob,
        continuous_score=ai_prob,
        explanation=(
            "Phát hiện dấu hiệu ảnh do AI tạo (detector tổng quát), "
            f"điểm {ai_prob:.3f}. Không tìm thấy khuôn mặt nên mô hình "
            "deepfake khuôn mặt (X2DFD) không áp dụng."
        ),
    )
    return ImageAnalyzeResponse(
        request_id=request_id,
        router=None,
        expert=None,
        final=final,
        ai_detector=AIDetectorResult(
            score=ai_prob,
            threshold=float(settings.AI_GATE_THRESHOLD),
            model=Path(settings.AI_GATE_MODEL).name,
            overridden=True,
        ),
        timing_ms=timing_ms,
        provenance=provenance,
    )
