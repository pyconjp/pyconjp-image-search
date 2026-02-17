"""Project-wide configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(os.environ.get("PYCONJP_PROJECT_ROOT", Path.cwd()))

load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = PROJECT_ROOT / "pyconjp_image_search.duckdb"
DATA_DIR = PROJECT_ROOT / "data" / "pyconjp"

# Flickr API
FLICKR_API_KEY = os.environ.get("FLICKR_API_KEY", "")
FLICKR_API_BASE = "https://api.flickr.com/services/rest/"
FLICKR_USER_ID = os.environ.get("FLICKR_USER_ID", "")

# Embedding – SigLIP 2 base
SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"
EMBEDDING_DIM = 768
EMBEDDING_IMAGE_SIZE = 224

# Embedding – SigLIP 2 Large
SIGLIP_LARGE_MODEL_NAME = "google/siglip2-large-patch16-256"
SIGLIP_LARGE_DB_PATH = PROJECT_ROOT / "pyconjp_image_search_siglip2_large.duckdb"
SIGLIP_LARGE_EMBEDDING_DIM = 1024
SIGLIP_LARGE_IMAGE_SIZE = 256

# Embedding – CLIP-L
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_DB_PATH = PROJECT_ROOT / "pyconjp_image_search_clip.duckdb"
CLIP_EMBEDDING_DIM = 768

# Face detection – InsightFace
INSIGHTFACE_MODEL_NAME = "insightface/buffalo_l"
FACE_EMBEDDING_DIM = 512

# Model configurations lookup
MODEL_CONFIGS: dict[str, dict] = {
    "siglip": {
        "model_name": SIGLIP_MODEL_NAME,
        "db_path": DB_PATH,
        "embedding_dim": EMBEDDING_DIM,
    },
    "siglip-large": {
        "model_name": SIGLIP_LARGE_MODEL_NAME,
        "db_path": SIGLIP_LARGE_DB_PATH,
        "embedding_dim": SIGLIP_LARGE_EMBEDDING_DIM,
    },
    "clip": {
        "model_name": CLIP_MODEL_NAME,
        "db_path": CLIP_DB_PATH,
        "embedding_dim": CLIP_EMBEDDING_DIM,
    },
}
