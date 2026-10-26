from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from web.backend.app.config import settings
from web.backend.app.utils.logger import logger

REAL_LABELS = {"real", "human"}


class AIImageDetector:
    """General AI-generated-image classifier (HF image classification model, loaded once, kept warm)."""

    def __init__(self, model_dir: str = ""):
        self.model_dir = model_dir or settings.AI_GATE_MODEL
        self.model = None
        self.processor = None
        self.device = "cpu"
        self._ai_indices = []

    def _pick_device(self) -> str:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        free_bytes, _ = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024 ** 3)
        if free_gb > settings.AI_GATE_MIN_FREE_GPU_GB:
            return "cuda:0"
        logger.warning("AI detector: only %.1f GB GPU free, using CPU", free_gb)
        return "cpu"

    def load_model(self):
        if self.model is not None:
            return

        model_dir = Path(self.model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"AI detector model dir does not exist: '{model_dir}'")

        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = self._pick_device()
        logger.info(f"Loading AI image detector on {self.device} from {model_dir}")
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True)
        self.model = model.to(self.device).eval()
        self._ai_indices = [
            int(i) for i, label in model.config.id2label.items()
            if label.lower() not in REAL_LABELS
        ]
        logger.info("AI image detector loaded (AI classes: %s)", self._ai_indices)

    def score(self, image_path: Path) -> float:
        """Probability that the full image is AI-generated."""
        self.load_model()
        import torch

        with Image.open(image_path) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")

        inputs = self.processor(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(pixel_values=inputs).logits, dim=-1)[0]
        return float(probs[self._ai_indices].sum().clamp(0.0, 1.0))

    def try_score(self, image_path: Optional[Path]) -> Optional[float]:
        """Score, or None if disabled/unavailable. Never breaks the X2DFD pipeline."""
        if not settings.AI_GATE_ENABLED or image_path is None:
            return None
        try:
            return self.score(image_path)
        except Exception as exc:
            logger.warning("AI image detector unavailable, keeping X2DFD verdict: %s", exc)
            return None
