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

    def test_symlinked_spelling_of_same_file_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            real_dir = Path(td) / "real"
            real_dir.mkdir()
            (real_dir / "a.png").touch()
            link_dir = Path(td) / "link"
            link_dir.symlink_to(real_dir)
            out = Path(td) / "scores.csv"
            with out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image_path", "score"])
                writer.writeheader()
                # wrapper echoes the path through the symlink
                writer.writerow({"image_path": str(link_dir / "a.png"), "score": "0.7"})

            scores = ExpertClient._read_scores(out, [real_dir / "a.png"])
            self.assertEqual(list(scores.values()), [0.7])


if __name__ == "__main__":
    unittest.main()
