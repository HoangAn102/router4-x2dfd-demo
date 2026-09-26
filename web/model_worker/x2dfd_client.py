import asyncio
import json
import math
import os

from pathlib import Path
from typing import Any, Optional, Tuple

from web.backend.app.config import settings
from web.backend.app.schemas.image import FinalResult
from web.backend.app.utils.logger import logger


class FinalScoreUnavailableError(Exception):
    pass


class X2DFDClient:

    # Classification follows the short X2DFD score path.
    SCORE_MAX_NEW_TOKENS = 4

    # Original X2DFD single_image_infer() default.
    ORIGINAL_EXPLANATION_MAX_NEW_TOKENS = 512

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

        self._proc = None
        self._request_lock = asyncio.Lock()


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
            (int, float),
        ) or not isinstance(
            fake_score,
            (int, float),
        ):
            raise FinalScoreUnavailableError(
                "REAL/FAKE score missing/non-numeric"
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
            0 <= real <= 1
            and 0 <= fake <= 1
        ):
            raise FinalScoreUnavailableError(
                "Final score outside [0,1]"
            )

        if abs(real + fake - 1.0) > 1e-4:
            raise FinalScoreUnavailableError(
                "REAL/FAKE pair not normalized"
            )

        return real, fake




    async def stop_worker(self):

        proc = self._proc
        self._proc = None

        if proc is None:
            return

        if proc.returncode is None:

            try:
                proc.terminate()

                await asyncio.wait_for(
                    proc.wait(),
                    timeout=5,
                )

            except Exception:

                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass


    async def _ensure_worker(self):

        if (
            self._proc is not None
            and self._proc.returncode is None
        ):
            return

        runtime = Path(
            settings.X2PYTHON_BIN
        )

        if not runtime.exists():
            raise FinalScoreUnavailableError(
                "X2DFD runtime missing"
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

        worker_script = (
            Path(__file__)
            .resolve()
            .parent
            / "x2dfd_worker_cli.py"
        )

        env = dict(os.environ)

        env["PYTHONPATH"] = (
            settings.X2DFD_PROJECT_ROOT
        )

        env["X2DFD_PROJECT_ROOT"] = (
            settings.X2DFD_PROJECT_ROOT
        )

        logger.info(
            "Starting persistent X2DFD worker"
        )

        self._proc = (
            await asyncio.create_subprocess_exec(
                settings.X2PYTHON_BIN,

                str(worker_script),

                "--server",

                "--lora-dir",
                self.lora_dir,

                "--base-model",
                self.base_model,

                "--max-new-tokens",
                str(self.SCORE_MAX_NEW_TOKENS),

                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,

                # inherited by Uvicorn -> backend logfile
                stderr=None,

                cwd=settings.X2DFD_PROJECT_ROOT,
                env=env,
            )
        )


    async def _request(
        self,
        payload: dict,
    ) -> dict:

        async with self._request_lock:

            for attempt in range(2):

                await self._ensure_worker()

                proc = self._proc

                try:

                    if (
                        proc.stdin is None
                        or proc.stdout is None
                    ):
                        raise RuntimeError(
                            "X2DFD worker pipes unavailable"
                        )

                    wire = (
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode()

                    proc.stdin.write(wire)
                    await proc.stdin.drain()

                    raw = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=settings.IMAGE_INFERENCE_TIMEOUT_SEC,
                    )

                    if not raw:
                        raise BrokenPipeError(
                            "X2DFD worker closed stdout"
                        )

                    return json.loads(
                        raw.decode("utf-8")
                    )

                except (
                    BrokenPipeError,
                    ConnectionResetError,
                ):

                    await self.stop_worker()

                    if attempt == 1:
                        raise

                except asyncio.TimeoutError:

                    await self.stop_worker()

                    raise FinalScoreUnavailableError(
                        "X2DFD persistent worker timed out"
                    )

            raise FinalScoreUnavailableError(
                "X2DFD persistent worker unavailable"
            )


    async def infer(
        self,
        image_path: Path,
        selected_alias: str,
        calibrated_score: float,
        *,
        include_explanation: bool = True,
    ) -> FinalResult:

        image_path = (
            Path(image_path)
            .expanduser()
            .resolve(strict=True)
        )

        if not image_path.is_file():
            raise FinalScoreUnavailableError(
                f"Input image missing: {image_path}"
            )

        prompt = self.build_wfs_prompt(
            selected_alias,
            calibrated_score,
        )

        # ----------------------------------------------------
        # A. SCIENTIFIC SCORE
        # ----------------------------------------------------

        response = await self._request(
            {
                "mode": "score",
                "image": str(image_path),
                "prompt": prompt,
                "lora_dir": self.lora_dir,
                "base_model": self.base_model,
                "max_new_tokens":
                    self.SCORE_MAX_NEW_TOKENS,
            }
        )

        if not response.get("ok"):

            raise FinalScoreUnavailableError(
                "X2DFD score worker failed: "
                f"{response.get('error_type', 'Error')}: "
                f"{response.get('error', 'unknown')}"
            )

        real, fake = self.validate_scores(
            response.get("real_score"),
            response.get("fake_score"),
        )

        verdict = (
            "FAKE"
            if fake >= 0.5
            else "REAL"
        )


        # ----------------------------------------------------
        # B. ORIGINAL X2DFD EXPLANATION
        #
        # Returned exactly as single_image_infer() produces it.
        # Web layer does not modify the model's text.
        # ----------------------------------------------------

        explanation = None

        if include_explanation:

            try:

                exp_response = (
                    await self._request(
                        {
                            "mode":
                                "explain",

                            "image":
                                str(
                                    image_path
                                ),

                            "prompt":
                                prompt,

                            "lora_dir":
                                self.lora_dir,

                            "base_model":
                                self.base_model,

                            "max_new_tokens":
                                self.ORIGINAL_EXPLANATION_MAX_NEW_TOKENS,
                        }
                    )
                )

                if exp_response.get("ok"):

                    # NO post-processing.
                    explanation = (
                        exp_response.get(
                            "answer"
                        )
                    )

                else:

                    logger.warning(
                        "Original X2DFD explanation failed: %s",
                        exp_response.get(
                            "error",
                            "unknown",
                        ),
                    )

            except Exception as exc:

                # Explanation is optional presentation.
                # Genuine detector score remains valid.
                logger.warning(
                    "Original X2DFD explanation unavailable: %s",
                    exc,
                )

        logger.info(
            "FINAL X2DFD: "
            "real=%.8f fake=%.8f verdict=%s",
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
