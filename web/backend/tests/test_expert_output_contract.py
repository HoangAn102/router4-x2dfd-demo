import csv
import tempfile
import unittest
from pathlib import Path

from web.model_worker.expert_client import ExpertClient, ExpertInferenceError


class ExpertOutputContractTests(unittest.TestCase):
    def test_duplicate_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "scores.csv"
            image_path = Path(td) / "a.png"
            image_path.touch()
            image = str(image_path.resolve())
            with out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image_path", "score"])
                writer.writeheader()
                writer.writerow({"image_path": image, "score": "0.2"})
                writer.writerow({"image_path": image, "score": "0.8"})

            with self.assertRaisesRegex(ExpertInferenceError, "duplicate"):
                ExpertClient._read_scores(out, [Path(image)])

    def test_unexpected_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "scores.csv"
            expected_path = Path(td) / "expected.png"
            expected_path.touch()
            expected = str(expected_path.resolve())
            unexpected = str((Path(td) / "other.png").resolve())
            with out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image_path", "score"])
                writer.writeheader()
                writer.writerow({"image_path": unexpected, "score": "0.2"})

            with self.assertRaisesRegex(ExpertInferenceError, "unexpected"):
                ExpertClient._read_scores(out, [Path(expected)])


if __name__ == "__main__":
    unittest.main()
