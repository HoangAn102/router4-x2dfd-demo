"""Calibrator used live must be the one that produced the LoRA's SFT prompt scores (audit 2026-10-30, P3)."""
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.isotonic import IsotonicRegression

from web.backend.app.config import settings
from web.model_worker.expert_client import ExpertClient

EXPERTS = ["blending", "diffusion", "frequency", "texture"]


def _iso(y_hi):
    return IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0.1, y_hi])


class CalibratorBundleTests(unittest.TestCase):
    def _load(self, bundle):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.joblib"
            joblib.dump(bundle, p)
            ec = ExpertClient(str(p))
            ec.load_calibrators()
            return ec

    def test_v2_bundle_uses_calibrators_key(self):
        # router_teacher_v2: "experts" is a list of names, the models live under "calibrators"
        ec = self._load({"experts": EXPERTS, "temperature": 0.2,
                         "calibrators": {e: _iso(0.9) for e in EXPERTS}})
        self.assertAlmostEqual(ec.calibrate("texture", 1.0), 0.9)

    def test_final_clean_bundle_still_loads(self):
        ec = self._load({"experts": {e: _iso(0.8) for e in EXPERTS}})
        self.assertAlmostEqual(ec.calibrate("blending", 1.0), 0.8)

    def test_default_is_the_sft_calibrator(self):
        self.assertTrue(settings.CALIBRATORS_PATH.endswith("router_teacher_v2/calibrators_train_only.joblib"))


@unittest.skipUnless(Path(settings.CALIBRATORS_PATH).exists(), "server artifact not present")
class LiveCalibratorReproducesSftScores(unittest.TestCase):
    def test_known_sft_rows(self):
        # (expert, raw score, score printed in the SFT prompt), rows of
        # TRAIN24K_QC_20260914/train24k_manifest.csv. calibrators_FINAL_CLEAN gives 0.677/0.734/0.500/0.820.
        rows = [
            ("diffusion", 0.0023620743304491, 0.629),
            ("frequency", 6.600843335036188e-06, 0.560),
            ("texture", 6.904064662194287e-07, 0.291),
            ("blending", 0.8189746737480164, 0.856),
        ]
        ec = ExpertClient()
        for expert, raw, sft in rows:
            self.assertLess(abs(ec.calibrate(expert, raw) - sft), 0.005, expert)
