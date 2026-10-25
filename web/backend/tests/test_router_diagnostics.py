import tempfile
import unittest
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from web.model_worker.router_client import RouterClient


class _IdentityTransform:
    def __call__(self, _image):
        return torch.zeros(3, 224, 224)


class _FakeRouter:
    def __call__(self, _tensor):
        return torch.tensor([[0.1, 0.2, 2.0, 0.0]], dtype=torch.float32)


class RouterDiagnosticsTests(unittest.TestCase):
    def test_route_records_vector_without_changing_selection(self):
        with tempfile.TemporaryDirectory() as td:
            image = Path(td) / "case.png"
            Image.new("RGB", (224, 224), (10, 20, 30)).save(image)

            client = RouterClient(checkpoint_path=str(image))
            client.model = _FakeRouter()
            client.device = "cpu"
            client._transform = _IdentityTransform()

            selected, alias, confidence, margin = client.route_image(image)

            self.assertEqual(selected, "frequency")
            self.assertEqual(alias, "Frequency")
            self.assertGreater(confidence, 0.0)
            self.assertGreaterEqual(margin, 0.0)
            self.assertEqual(client.last_route_diagnostics["experts"], [
                "blending", "diffusion", "frequency", "texture"
            ])
            self.assertEqual(client.last_route_diagnostics["selected_index"], 2)
            self.assertEqual(len(client.last_route_diagnostics["logits"]), 4)
            self.assertEqual(len(client.last_route_diagnostics["probabilities"]), 4)


if __name__ == "__main__":
    unittest.main()
