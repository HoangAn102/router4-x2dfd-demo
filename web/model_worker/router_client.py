import math
from pathlib import Path
from typing import Dict, List, Literal, Tuple
from PIL import Image
from web.backend.app.config import settings
from web.backend.app.utils.logger import logger

EXPERTS = ["blending", "diffusion", "frequency", "texture"]
ALIASES = {
    "blending": "Blending",
    "diffusion": "Diffusion",
    "frequency": "Frequency",
    "texture": "Texture",
}


class RouterClient:
    """Router4 client loading EfficientNet-B0 and classifying image into one of 4 experts."""

    def __init__(self, checkpoint_path: str = ""):
        self.checkpoint_path = checkpoint_path or settings.ROUTER_CHECKPOINT
        self.model = None
        self.device = "cpu"
        self._transform = None

    def _init_transform(self):
        from torchvision import transforms  # type: ignore
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_model(self):
        """Lazy loader for PyTorch EfficientNet-B0 model."""
        if self.model is not None:
            return

        import torch
        import torch.nn as nn
        from torchvision.models import efficientnet_b0  # type: ignore

        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Router checkpoint does not exist at '{self.checkpoint_path}'"
            )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Router4 model on {self.device} from {ckpt_path.name}")

        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(EXPERTS))

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()

        self.model = model
        self._transform = self._init_transform()
        logger.info("Router4 model loaded successfully.")

    def route_image(
        self, image_path: Path
    ) -> Tuple[Literal["blending", "diffusion", "frequency", "texture"], str, float, float]:
        """
        Routes an image to exactly 1 expert out of {blending, diffusion, frequency, texture}.
        Returns: (selected_expert, selected_alias, router_confidence, margin)
        """
        self.load_model()
        import torch

        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            tensor = self._transform(rgb_img).unsqueeze(0).to(self.device)  # type: ignore

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            top2 = probs.topk(2, dim=1)

            top1_idx = int(top2.indices[0, 0].item())
            top1_conf = float(top2.values[0, 0].item())
            top2_conf = float(top2.values[0, 1].item())
            margin = float(top1_conf - top2_conf)

        selected_expert = EXPERTS[top1_idx]
        selected_alias = ALIASES[selected_expert]

        logger.info(
            f"Router4 decision for {image_path.name}: {selected_expert} "
            f"(confidence={top1_conf:.4f}, margin={margin:.4f})"
        )
        return selected_expert, selected_alias, round(top1_conf, 4), round(margin, 4)  # type: ignore
