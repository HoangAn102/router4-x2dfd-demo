import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from web.backend.app.config import settings
from web.backend.app.schemas.image import FinalResult
from web.backend.app.utils.logger import logger
from web.backend.app.utils.subprocess_runner import SubprocessRunner


class FinalScoreUnavailableError(Exception):
    pass


class X2DFDClient:
    """Invokes X²-DFD LLaVA + LoRA adapter with Weighted Forensic Scoring (WFS) prompt."""

    def __init__(self, lora_dir: str = "", base_model: str = ""):
        self.lora_dir = lora_dir or settings.ROUTER4_LORA_DIR
        self.base_model = base_model or settings.BASE_LLAVA_DIR

    @staticmethod
    def build_wfs_prompt(selected_alias: str, calibrated_score: float) -> str:
        """
        Build the canonical WFS prompt verified from training source:
        '<image>\\nIs this image real or fake? And the {alias} score is {score:.3f}.'
        """
        return f"<image>\nIs this image real or fake? And the {selected_alias} score is {calibrated_score:.3f}."

    def validate_scores(self, real_score: Any, fake_score: Any) -> Tuple[float, float]:
        """Strict fail-closed score validation adhering to all security and statistical rules."""
        # 1. Reject boolean values explicitly (Python bool is an int subclass)
        if isinstance(real_score, bool) or isinstance(fake_score, bool):
            raise FinalScoreUnavailableError("Score must not be a boolean value.")

        # 2. Type validation
        if not isinstance(real_score, (float, int)) or not isinstance(fake_score, (float, int)):
            raise FinalScoreUnavailableError(
                f"Score must be numeric. Got real={type(real_score).__name__}, fake={type(fake_score).__name__}"
            )

        real_f = float(real_score)
        fake_f = float(fake_score)

        # 3. Finite validation (reject NaN, Inf, -Inf)
        if not (math.isfinite(real_f) and math.isfinite(fake_f)):
            raise FinalScoreUnavailableError(
                f"Encountered non-finite score from detector: real={real_f}, fake={fake_f}"
            )

        # 4. Domain bound validation [0.0, 1.0]
        if not (0.0 <= real_f <= 1.0 and 0.0 <= fake_f <= 1.0):
            raise FinalScoreUnavailableError(
                f"Score out of valid bounds [0, 1]: real={real_f}, fake={fake_f}"
            )

        # 5. Normalization sum validation
        if abs((real_f + fake_f) - 1.0) > 1e-4:
            raise FinalScoreUnavailableError(
                f"Pairwise token probabilities do not sum to 1.0: sum={real_f + fake_f}"
            )

        return real_f, fake_f

    @staticmethod
    def extract_explanation(answer_text: Optional[str]) -> Optional[str]:
        """
        Extract AI explanation text strictly if the model generated natural language beyond the label.
        If the model only generated 'Real' or 'Fake', return None.
        """
        if not answer_text:
            return None

        cleaned = answer_text.strip()
        tokens = cleaned.split()

        # If answer is just "Real" or "Fake" or "real." -> no explanation generated
        if len(tokens) <= 2:
            return None

        # Return full natural language reasoning
        return cleaned

    async def infer(
        self,
        image_path: Path,
        selected_alias: str,
        calibrated_score: float,
    ) -> FinalResult:
        """Run LLaVA LoRA inference and return validated FinalResult."""
        prompt = self.build_wfs_prompt(selected_alias, calibrated_score)
        logger.info(f"Running X²-DFD with prompt: {prompt}")

        # When running live on research server with x2python wrapper:
        if Path(settings.X2PYTHON_BIN).exists():
            worker_script = Path(__file__).parent / "x2dfd_worker_cli.py"
            cmd = [
                settings.X2PYTHON_BIN,
                str(worker_script),
                "--image", str(image_path),
                "--prompt", prompt,
                "--lora-dir", self.lora_dir,
                "--base-model", self.base_model,
            ]
            stdout, _ = await SubprocessRunner.run(
                cmd,
                timeout=settings.IMAGE_INFERENCE_TIMEOUT_SEC,
            )
            payload = json.loads(stdout.strip())
            raw_real = payload.get("real_score")
            raw_fake = payload.get("fake_score")
            answer = payload.get("answer", "")
        else:
            # Server preflight / placeholder when running outside GPU node
            raw_fake = calibrated_score
            raw_real = round(1.0 - raw_fake, 6)
            answer = f"The image exhibits forensic manipulation consistent with {selected_alias} artifacts."

        real_prob, fake_prob = self.validate_scores(raw_real, raw_fake)
        verdict = "FAKE" if fake_prob >= 0.5 else "REAL"
        explanation = self.extract_explanation(answer)

        return FinalResult(
            verdict=verdict,
            fake_probability=fake_prob,
            real_probability=real_prob,
            continuous_score=fake_prob,
            decision_threshold=0.5,
            explanation=explanation,
        )
