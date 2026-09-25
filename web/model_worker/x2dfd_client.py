import json
import math
from pathlib import Path
from typing import Any, Optional, Tuple

from web.backend.app.config import settings
from web.backend.app.schemas.image import FinalResult
from web.backend.app.utils.logger import logger
from web.backend.app.utils.subprocess_runner import SubprocessRunner


class FinalScoreUnavailableError(Exception):
    pass


class X2DFDClient:
    """LIVE X²-DFD + Router-aware LoRA client."""

    def __init__(
        self,
        lora_dir: str = "",
        base_model: str = "",
    ):
        self.lora_dir = (
            lora_dir
            or settings.ROUTER4_LORA_DIR
        )

        self.base_model = (
            base_model
            or settings.BASE_LLAVA_DIR
        )

    @staticmethod
    def build_wfs_prompt(
        selected_alias: str,
        calibrated_score: float,
    ) -> str:

        return (
            "<image>\n"
            "Is this image real or fake? "
            f"And the {selected_alias} score is "
            f"{calibrated_score:.3f}."
        )

    @staticmethod
    def validate_scores(
        real_score: Any,
        fake_score: Any,
    ) -> Tuple[float, float]:

        if (
            isinstance(real_score, bool)
            or isinstance(fake_score, bool)
        ):
            raise FinalScoreUnavailableError(
                "REAL/FAKE score cannot be bool"
            )

        if not isinstance(
            real_score,
            (float, int),
        ) or not isinstance(
            fake_score,
            (float, int),
        ):
            raise FinalScoreUnavailableError(
                "REAL/FAKE score is missing/non-numeric"
            )

        real = float(real_score)
        fake = float(fake_score)

        if not (
            math.isfinite(real)
            and math.isfinite(fake)
        ):
            raise FinalScoreUnavailableError(
                "Non-finite final score"
            )

        if not (
            0.0 <= real <= 1.0
            and 0.0 <= fake <= 1.0
        ):
            raise FinalScoreUnavailableError(
                "Final score outside [0,1]"
            )

        if abs(
            (real + fake) - 1.0
        ) > 1e-4:
            raise FinalScoreUnavailableError(
                "REAL/FAKE pair not normalized"
            )

        return real, fake

    @staticmethod
    def extract_explanation(
        answer_text: Optional[str],
    ) -> Optional[str]:

        if not answer_text:
            return None

        cleaned = answer_text.strip()

        if len(cleaned.split()) <= 2:
            return None

        return cleaned

    async def infer(
        self,
        image_path: Path,
        selected_alias: str,
        calibrated_score: float,
    ) -> FinalResult:

        if not Path(
            settings.X2PYTHON_BIN
        ).exists():
            raise FinalScoreUnavailableError(
                "X²-DFD runtime missing. "
                "Refusing expert-score substitution."
            )

        if not Path(
            self.lora_dir
        ).exists():
            raise FinalScoreUnavailableError(
                "Router-aware LoRA missing"
            )

        if not Path(
            self.base_model
        ).exists():
            raise FinalScoreUnavailableError(
                "Base LLaVA missing"
            )

        prompt = self.build_wfs_prompt(
            selected_alias,
            calibrated_score,
        )

        worker_script = (
            Path(__file__).parent
            / "x2dfd_worker_cli.py"
        )

        stdout, _ = (
            await SubprocessRunner.run(
                [
                    settings.X2PYTHON_BIN,
                    str(worker_script),
                    "--image",
                    str(image_path),
                    "--prompt",
                    prompt,
                    "--lora-dir",
                    self.lora_dir,
                    "--base-model",
                    self.base_model,
                    "--max-new-tokens",
                    "32",
                ],
                timeout=settings.IMAGE_INFERENCE_TIMEOUT_SEC,
                cwd=settings.X2DFD_PROJECT_ROOT,
                custom_env={
                    "PYTHONPATH":
                        settings.X2DFD_PROJECT_ROOT,

                    "X2DFD_PROJECT_ROOT":
                        settings.X2DFD_PROJECT_ROOT,
                },
            )
        )

        payload = None

        # Model libraries may log before JSON.
        # Find the LAST valid score payload.
        for line in reversed(
            stdout.splitlines()
        ):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            if (
                isinstance(obj, dict)
                and "real_score" in obj
                and "fake_score" in obj
            ):
                payload = obj
                break

        if payload is None:
            raise FinalScoreUnavailableError(
                "No valid final-score JSON "
                "returned by X²-DFD"
            )

        real, fake = self.validate_scores(
            payload.get("real_score"),
            payload.get("fake_score"),
        )

        verdict = (
            "FAKE"
            if fake >= 0.5
            else "REAL"
        )

        explanation = (
            self.extract_explanation(
                payload.get("answer", "")
            )
        )

        logger.info(
            "FINAL X2DFD: real=%.8f fake=%.8f verdict=%s",
            real,
            fake,
            verdict,
        )

        return FinalResult(
            verdict=verdict,
            fake_probability=fake,
            real_probability=real,
            continuous_score=fake,
            decision_threshold=0.5,
            explanation=explanation,
        )
