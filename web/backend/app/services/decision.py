"""
Single decision function shared by the web workers and the evaluation scripts.

Order (first gate that fires wins; the X2DFD verdict is the default):
  1. AI-image gate     ai_prob >= AI_GATE_THRESHOLD        -> FAKE (general AI-generated image)
  2. Blending gate     raw >= BLENDING_GATE_THRESHOLD      -> FAKE, only when still REAL
  3. X2DFD verdict     fake_probability >= 0.5 (set in X2DFDClient.infer)
Borderline AI scores (AI_GATE_WARN <= ai_prob < threshold) only lower reliability.
Both gates fail open: a None score leaves the X2DFD result untouched.
"""

from typing import Optional, Tuple

from ..config import settings
from ..schemas.image import AIDetectorResult, BlendingGateResult, FinalResult, Reliability
from .ai_gate import apply_ai_gate


def apply_blending_gate(
    final: FinalResult,
    blending_raw: Optional[float],
    already_overridden: bool,
) -> Tuple[FinalResult, Optional[BlendingGateResult]]:
    if blending_raw is None or not settings.BLENDING_GATE_ENABLED:
        return final, None

    raw = min(max(float(blending_raw), 0.0), 1.0)
    gate = BlendingGateResult(raw=raw, threshold=float(settings.BLENDING_GATE_THRESHOLD))

    if raw >= gate.threshold and final.verdict == "REAL" and not already_overridden:
        gate.overridden = True
        gate.x2dfd_fake_probability = final.fake_probability
        final = final.model_copy(update={
            "verdict": "FAKE",
            "fake_probability": raw,
            "real_probability": 1.0 - raw,
            "continuous_score": raw,
            "explanation": (
                "Phát hiện dấu vết ghép/hòa trộn khuôn mặt (expert blending), "
                f"điểm {raw:.3f}."
            ),
        })

    return final, gate


def decide(
    final: FinalResult,
    reliability: Reliability,
    ai_prob: Optional[float],
    blending_raw: Optional[float],
) -> Tuple[FinalResult, Reliability, Optional[AIDetectorResult], Optional[BlendingGateResult]]:
    """Combine the X2DFD result with the AI-image gate and the blending gate."""

    final, reliability, ai_detector = apply_ai_gate(final, reliability, ai_prob)
    final, blending_gate = apply_blending_gate(
        final,
        blending_raw,
        already_overridden=bool(ai_detector and ai_detector.overridden),
    )
    return final, reliability, ai_detector, blending_gate
