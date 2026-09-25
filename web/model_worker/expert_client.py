import csv
import math
import tempfile
from pathlib import Path
from typing import Literal, Tuple

from web.backend.app.config import settings
from web.backend.app.utils.logger import logger
from web.backend.app.utils.subprocess_runner import SubprocessRunner


class ExpertInferenceError(RuntimeError):
    pass


class ExpertClient:
    """
    LIVE dispatcher for the frozen Router4 expert pool.

    Scientific invariants:
      - exact selected expert only
      - reuse research benchmark wrappers
      - no placeholder/fallback score
      - expert-specific calibration
      - failure => fail closed
    """

    def __init__(self, calibrators_path: str = ""):
        self.calibrators_path = (
            calibrators_path or settings.CALIBRATORS_PATH
        )
        self.calibrators = None

    def load_calibrators(self):
        if self.calibrators is not None:
            return

        import joblib

        path = Path(self.calibrators_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Calibrator bundle missing: {path}"
            )

        bundle = joblib.load(path)

        if isinstance(bundle, dict) and "experts" in bundle:
            self.calibrators = bundle["experts"]
        else:
            self.calibrators = bundle

        required = {
            "blending",
            "diffusion",
            "frequency",
            "texture",
        }

        missing = required - set(self.calibrators.keys())

        if missing:
            raise ExpertInferenceError(
                f"Missing calibrators: {sorted(missing)}"
            )

    def calibrate(
        self,
        expert: str,
        raw_score: float,
    ) -> float:

        self.load_calibrators()

        calibrated = float(
            self.calibrators[expert]
            .predict([float(raw_score)])[0]
        )

        if not math.isfinite(calibrated):
            raise ExpertInferenceError(
                f"Non-finite calibrated score: {calibrated}"
            )

        if not 0.0 <= calibrated <= 1.0:
            raise ExpertInferenceError(
                f"Calibrated score outside [0,1]: {calibrated}"
            )

        return round(calibrated, 8)

    @staticmethod
    def _write_manifest(
        output_dir: Path,
        image_path: Path,
    ):
        manifest = output_dir / "manifest.csv"

        with manifest.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=["image_path"],
            )

            writer.writeheader()

            writer.writerow({
                "image_path": str(image_path.resolve())
            })

    @staticmethod
    def _read_score(
        csv_path: Path,
        image_path: Path,
    ) -> float:

        if not csv_path.exists():
            raise ExpertInferenceError(
                f"Expert output missing: {csv_path}"
            )

        with csv_path.open(
            newline="",
            encoding="utf-8",
        ) as f:
            rows = list(csv.DictReader(f))

        target = str(image_path.resolve())

        for row in rows:
            if row.get("image_path") == target:
                value = float(row["score"])

                if not math.isfinite(value):
                    raise ExpertInferenceError(
                        f"Non-finite raw score: {value}"
                    )

                return value

        if len(rows) == 1:
            value = float(rows[0]["score"])

            if math.isfinite(value):
                return value

        raise ExpertInferenceError(
            f"No expert score for {target}"
        )

    async def compute_expert_score(
        self,
        expert: Literal[
            "blending",
            "diffusion",
            "frequency",
            "texture",
        ],
        image_path: Path,
    ) -> Tuple[float, float]:

        raw = await self._run_expert_inference(
            expert,
            image_path,
        )

        calibrated = self.calibrate(
            expert,
            raw,
        )

        logger.info(
            "Expert %s: raw=%.8f calibrated=%.8f",
            expert,
            raw,
            calibrated,
        )

        return raw, calibrated

    async def _run_expert_inference(
        self,
        expert: str,
        image_path: Path,
    ) -> float:

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        wrappers = Path(
            settings.ROUTER8_WRAPPER_ROOT
        )

        with tempfile.TemporaryDirectory(
            prefix=f"safevision_{expert}_"
        ) as tmp:

            out = Path(tmp)

            self._write_manifest(
                out,
                image_path,
            )

            if expert in {
                "blending",
                "diffusion",
            }:

                wrapper = wrappers / "run_x2.py"

                cmd = [
                    settings.X2PYTHON_BIN,
                    str(wrapper),
                    "--out",
                    str(out),
                    "--expert",
                    expert,
                ]

                cwd = settings.X2DFD_PROJECT_ROOT

                csv_path = (
                    out
                    / f"{expert}_scores.csv"
                )

            elif expert == "frequency":

                wrapper = (
                    wrappers
                    / "run_frequency.py"
                )

                cmd = [
                    settings.FREQPYTHON_BIN,
                    str(wrapper),
                    "--out",
                    str(out),
                    "--name",
                    "SAFEVISION_WEB_SINGLE",
                ]

                cwd = settings.DFFREQ_PROJECT_ROOT

                csv_path = (
                    out / "frequency_scores.csv"
                )

            elif expert == "texture":

                wrapper = (
                    wrappers
                    / "run_texture.py"
                )

                cmd = [
                    settings.TEXPYTHON_BIN,
                    str(wrapper),
                    "--out",
                    str(out),
                    "--work",
                    settings.TEXTURE_PROJECT_ROOT,
                ]

                cwd = settings.TEXTURE_PROJECT_ROOT

                csv_path = (
                    out / "texture_scores.csv"
                )

            else:
                raise ExpertInferenceError(
                    f"Unknown expert: {expert}"
                )

            if not wrapper.exists():
                raise FileNotFoundError(wrapper)

            await SubprocessRunner.run(
                cmd,
                timeout=settings.EXPERT_INFERENCE_TIMEOUT_SEC,
                cwd=cwd,
                custom_env={
                    "PYTHONPATH": cwd,
                },
            )

            return self._read_score(
                csv_path,
                image_path,
            )
