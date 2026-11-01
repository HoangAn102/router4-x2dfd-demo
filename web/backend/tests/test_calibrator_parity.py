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
        # deployed lineage V3 (01/11): the calibrator that wrote the V3 SFT prompt scores, with the V3 LoRA
        self.assertTrue(settings.CALIBRATORS_PATH.endswith("ROUTER4_V3_20261030/01_calibrators/calibrators_V3.joblib"))
        self.assertTrue(settings.ROUTER4_LORA_DIR.endswith("ROUTER4_V3_20261030/06_lora/router4_x2dfd_V3_full30k"))


CAL_B = "/home/aiotlab/hoangan/outputs/router_teacher_v2/calibrators_train_only.joblib"
CAL_V3 = "/home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030/01_calibrators/calibrators_V3.joblib"


@unittest.skipUnless(Path(CAL_V3).exists(), "server artifact not present")
class LiveCalibratorReproducesV3SftScores(unittest.TestCase):
    def test_known_v3_sft_rows(self):
        # (expert, raw score from 02_teacher/train_teacher_V3.csv, score printed in router4_x2dfd_V3_SFT.json)
        rows = [("diffusion", 0.0106221856549382, 0.577), ("blending", 0.0028837039135396, 0.543),
                ("frequency", 1.0, 0.795), ("texture", 0.99968159198761, 1.0)]
        ec = ExpertClient(CAL_V3)
        for expert, raw, sft in rows:
            self.assertLess(abs(ec.calibrate(expert, raw) - sft), 0.005, expert)


@unittest.skipUnless(Path(CAL_B).exists(), "server artifact not present")
class LiveCalibratorReproducesSftScores(unittest.TestCase):
    def test_known_sft_rows(self):  # lineage (b)
        # (expert, raw score, score printed in the SFT prompt), rows of
        # TRAIN24K_QC_20260914/train24k_manifest.csv. calibrators_FINAL_CLEAN gives 0.677/0.734/0.500/0.820.
        rows = [
            ("diffusion", 0.0023620743304491, 0.629),
            ("frequency", 6.600843335036188e-06, 0.560),
            ("texture", 6.904064662194287e-07, 0.291),
            ("blending", 0.8189746737480164, 0.856),
        ]
        ec = ExpertClient(CAL_B)
        for expert, raw, sft in rows:
            self.assertLess(abs(ec.calibrate(expert, raw) - sft), 0.005, expert)
