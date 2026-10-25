import json
import tempfile
import unittest
from pathlib import Path

import pytest

write_failure = pytest.importorskip(
    "safevision_deep_audit_20261003.run_group_b_diagnostic"
).write_failure


class GroupBFailureArtifactTests(unittest.TestCase):
    def test_failure_artifact_preserves_exception_traceback(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "smoke.json.failure.json"
            try:
                raise RuntimeError("synthetic memory guard failure")
            except RuntimeError as exc:
                write_failure(out, "gpu_guard", exc)
            data = json.loads(out.read_text())
            self.assertEqual(data["status"], "FAILED")
            self.assertEqual(data["stage"], "gpu_guard")
            self.assertEqual(data["error_type"], "RuntimeError")
            self.assertIn("synthetic memory guard failure", data["traceback"])


if __name__ == "__main__":
    unittest.main()
