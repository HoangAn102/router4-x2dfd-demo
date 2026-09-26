import csv
import math
import tempfile

from pathlib import Path

from typing import (
    Dict,
    List,
    Literal,
    Tuple,
)

from web.backend.app.config import settings
from web.backend.app.utils.logger import logger
from web.backend.app.utils.subprocess_runner import SubprocessRunner


ExpertName = Literal[
    "blending",
    "diffusion",
    "frequency",
    "texture",
]


class ExpertInferenceError(RuntimeError):
    pass


class ExpertClient:

    def __init__(
        self,
        calibrators_path: str = "",
    ):

        self.calibrators_path = (
            calibrators_path
            or settings.CALIBRATORS_PATH
        )

        self.calibrators = None


    def load_calibrators(self):

        if self.calibrators is not None:
            return

        import joblib

        path = Path(
            self.calibrators_path
        )

        if not path.exists():
            raise FileNotFoundError(path)

        bundle = joblib.load(path)

        if (
            isinstance(bundle, dict)
            and "experts" in bundle
        ):
            self.calibrators = bundle["experts"]
        else:
            self.calibrators = bundle

        required = {
            "blending",
            "diffusion",
            "frequency",
            "texture",
        }

        missing = (
            required
            - set(self.calibrators.keys())
        )

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

        value = float(
            self.calibrators[
                expert
            ].predict(
                [float(raw_score)]
            )[0]
        )

        if not math.isfinite(value):
            raise ExpertInferenceError(
                "Non-finite calibrated score"
            )

        if not 0 <= value <= 1:
            raise ExpertInferenceError(
                "Calibrated score outside [0,1]"
            )

        return round(value, 8)


    @staticmethod
    def _write_manifest(
        out: Path,
        paths: List[Path],
    ):

        with (
            out / "manifest.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=["image_path"],
            )

            writer.writeheader()

            for p in paths:

                writer.writerow(
                    {
                        "image_path":
                            str(
                                Path(p)
                                .expanduser()
                                .resolve(strict=True)
                            )
                    }
                )


    @staticmethod
    def _read_scores(
        csv_path: Path,
        paths: List[Path],
    ) -> Dict[str, float]:

        if not csv_path.exists():
            raise ExpertInferenceError(
                f"Expert output missing: {csv_path}"
            )

        expected = [
            str(
                Path(p)
                .expanduser()
                .resolve(strict=True)
            )
            for p in paths
        ]

        found = {}

        with csv_path.open(
            newline="",
            encoding="utf-8",
        ) as f:

            for row in csv.DictReader(f):

                path = row.get("image_path")
                raw = row.get("score")

                if not path or raw in (None, ""):
                    continue

                score = float(raw)

                if not math.isfinite(score):
                    raise ExpertInferenceError(
                        "Non-finite expert score"
                    )

                found[path] = score

        missing = [
            p for p in expected
            if p not in found
        ]

        if missing:
            raise ExpertInferenceError(
                f"Expert produced {len(found)}/"
                f"{len(expected)} scores. "
                f"Missing: {missing[:3]}"
            )

        return {
            p: found[p]
            for p in expected
        }


    def _command(
        self,
        expert: ExpertName,
        out: Path,
    ):

        wrappers = Path(
            settings.ROUTER8_WRAPPER_ROOT
        )

        if expert in (
            "blending",
            "diffusion",
        ):

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
            csv_path = out / f"{expert}_scores.csv"


        elif expert == "frequency":

            wrapper = wrappers / "run_frequency.py"

            cmd = [
                settings.FREQPYTHON_BIN,
                str(wrapper),
                "--out",
                str(out),
                "--name",
                "SAFEVISION_VIDEO_BATCH",
            ]

            cwd = settings.DFFREQ_PROJECT_ROOT
            csv_path = out / "frequency_scores.csv"


        elif expert == "texture":

            wrapper = wrappers / "run_texture.py"

            cmd = [
                settings.TEXPYTHON_BIN,
                str(wrapper),
                "--out",
                str(out),
                "--work",
                settings.TEXTURE_PROJECT_ROOT,
            ]

            cwd = settings.TEXTURE_PROJECT_ROOT
            csv_path = out / "texture_scores.csv"


        else:
            raise ExpertInferenceError(
                f"Unknown expert: {expert}"
            )

        if not wrapper.exists():
            raise FileNotFoundError(wrapper)

        return cmd, cwd, csv_path


    async def compute_expert_scores_batch(
        self,
        expert: ExpertName,
        image_paths: List[Path],
    ) -> Dict[str, Tuple[float, float]]:

        if not image_paths:
            return {}

        paths = [
            Path(p)
            .expanduser()
            .resolve(strict=True)
            for p in image_paths
        ]

        with tempfile.TemporaryDirectory(
            prefix=f"safevision_{expert}_batch_"
        ) as tmp:

            out = Path(tmp)

            self._write_manifest(
                out,
                paths,
            )

            cmd, cwd, csv_path = (
                self._command(
                    expert,
                    out,
                )
            )

            logger.info(
                "Batch expert %s: %d frame(s)",
                expert,
                len(paths),
            )

            await SubprocessRunner.run(
                cmd,

                timeout=settings.EXPERT_INFERENCE_TIMEOUT_SEC,

                cwd=cwd,

                custom_env={
                    "PYTHONPATH": cwd,
                },
            )

            raw_scores = self._read_scores(
                csv_path,
                paths,
            )

        result = {}

        for p in paths:

            key = str(p)

            raw = raw_scores[key]

            calibrated = self.calibrate(
                expert,
                raw,
            )

            result[key] = (
                raw,
                calibrated,
            )

        return result


    async def compute_expert_score(
        self,
        expert: ExpertName,
        image_path: Path,
    ) -> Tuple[float, float]:

        p = (
            Path(image_path)
            .expanduser()
            .resolve(strict=True)
        )

        result = (
            await self.compute_expert_scores_batch(
                expert,
                [p],
            )
        )

        return result[str(p)]
