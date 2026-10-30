"""x2dfd_worker_cli one-shot mode (was calling _run_one with the wrong arguments)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model_worker"))
import x2dfd_worker_cli as cli  # noqa: E402


class OneShotTests(unittest.TestCase):
    def _args(self, image, mode="score"):
        return mock.Mock(image=str(image), prompt="\nIs this image real or fake? And the blending score is 0.300.",
                         lora_dir="L", base_model="B", max_new_tokens=4, mode=mode)

    def test_score_mode_prints_json(self):
        score = mock.Mock(return_value={"answer": "This image is real", "label_step": 3,
                                        "real_score": 0.9, "fake_score": 0.1})
        with tempfile.NamedTemporaryFile(suffix=".png") as f, \
                mock.patch.object(cli, "_load_infer_function", return_value=(score, mock.Mock())), \
                mock.patch("builtins.print") as pr:
            cli.oneshot_mode(self._args(f.name))
        out = json.loads(pr.call_args_list[-1].args[0])
        self.assertEqual((out["mode"], out["fake_score"]), ("score", 0.1))
        self.assertEqual(score.call_args.args[0], self._args(f.name).prompt)

    def test_explain_mode(self):
        explain = mock.Mock(return_value="This image is fake.")
        with tempfile.NamedTemporaryFile(suffix=".png") as f, \
                mock.patch.object(cli, "_load_infer_function", return_value=(mock.Mock(), explain)), \
                mock.patch("builtins.print") as pr:
            cli.oneshot_mode(self._args(f.name, mode="explain"))
        self.assertEqual(json.loads(pr.call_args_list[-1].args[0])["answer"], "This image is fake.")

    def test_missing_image_exits_nonzero(self):
        with mock.patch.object(cli, "_load_infer_function", return_value=(mock.Mock(), mock.Mock())):
            with self.assertRaises(SystemExit) as cm:
                cli.oneshot_mode(self._args("/nonexistent/x.png"))
        self.assertEqual(cm.exception.code, 1)

    def test_prompt_with_image_token_rejected(self):
        with self.assertRaises(ValueError):
            cli.build_train_prompt("<image>\nIs this image real or fake?")


if __name__ == "__main__":
    unittest.main()
