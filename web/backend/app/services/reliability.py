"""
Low-reliability flag for single-image results.

Presentation only: it never changes the selected expert, the scores or the
verdict, so the pipeline keeps the evaluated protocol
(Router4 top-1 -> calibrated expert score -> X2DFD, threshold 0.5).
"""

from ..config import settings
from ..schemas.image import Reliability


def assess_reliability(
    router_confidence: float,
    calibrated_score: float,
    llava_on_template: bool = True,
) -> Reliability:
    reasons = []

    low_conf = float(settings.ROUTER_LOW_CONFIDENCE)
    band = float(settings.EXPERT_NEUTRAL_BAND)

    if router_confidence < low_conf:
        reasons.append(
            "Router không chắc nên chọn chuyên gia nào "
            f"(độ tin cậy {router_confidence * 100:.1f}%, "
            f"ngưỡng {low_conf * 100:.0f}%)."
        )

    # round() avoids float noise at the band edges (0.6 - 0.5 = 0.0999...).
    if round(abs(calibrated_score - 0.5), 6) < band:
        reasons.append(
            "Điểm của chuyên gia được chọn gần mức trung tính "
            f"({calibrated_score:.2f}), gần như không cung cấp bằng chứng."
        )

    if not llava_on_template:
        reasons.append(
            "Mô hình ngôn ngữ không trả lời theo khuôn đã học; điểm được đọc tại vị trí nhãn "
            "của câu trả lời mẫu (\"This image is …\")."
        )

    return Reliability(
        level="low" if reasons else "normal",
        reasons=reasons,
    )
