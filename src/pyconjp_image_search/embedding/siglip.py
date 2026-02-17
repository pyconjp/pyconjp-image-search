"""SigLIP 2 model wrapper for image and text embedding."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from pyconjp_image_search.config import SIGLIP_MODEL_NAME


class SigLIPEmbedder:
    """Generate embeddings using SigLIP 2 model."""

    def __init__(
        self,
        model_name: str = SIGLIP_MODEL_NAME,
        device: str = "cuda",
    ) -> None:
        self.device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()  # type: ignore[arg-type]

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings."""
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / np.clip(norm, a_min=1e-8, a_max=None)

    @staticmethod
    def _extract_embeddings(outputs: object) -> np.ndarray:
        """Extract numpy embeddings from model output (tensor or BaseModelOutputWithPooling)."""
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output.float().cpu().numpy()  # type: ignore[union-attr]
        return outputs.float().cpu().numpy()  # type: ignore[union-attr]

    def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        """Embed a batch of images. Returns L2-normalized vectors."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        embeddings = self._extract_embeddings(outputs)
        return self._normalize(embeddings).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text query. Returns L2-normalized vector (1, dim)."""
        inputs = self.processor(
            text=[text], padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        embedding = self._extract_embeddings(outputs)
        return self._normalize(embedding).astype(np.float32)
