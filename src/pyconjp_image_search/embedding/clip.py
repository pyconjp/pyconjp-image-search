"""CLIP-L model wrapper for image and text embedding."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pyconjp_image_search.config import CLIP_MODEL_NAME


class CLIPEmbedder:
    """Generate embeddings using CLIP ViT-L/14 model."""

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()  # type: ignore[arg-type]

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings."""
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / np.clip(norm, a_min=1e-8, a_max=None)

    def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        """Embed a batch of images. Returns L2-normalized projected vectors (768-dim)."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")  # type: ignore[operator]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            # Explicitly run vision_model + visual_projection to get 768-dim
            # projected features. Do NOT use get_image_features() because some
            # transformers versions return BaseModelOutputWithPooling (1024-dim
            # pooler_output) instead of the projected tensor.
            vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            projected = self.model.visual_projection(vision_out.pooler_output)
        embeddings = projected.cpu().numpy()
        return self._normalize(embeddings).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text query. Returns L2-normalized projected vector (1, 768)."""
        proc_kwargs = dict(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = self.processor(**proc_kwargs)  # type: ignore[operator]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            # Explicitly run text_model + text_projection to guarantee
            # 768-dim projected features matching JS CLIPTextModelWithProjection.
            text_out = self.model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            projected = self.model.text_projection(text_out.pooler_output)
        embedding = projected.cpu().numpy()
        return self._normalize(embedding).astype(np.float32)
