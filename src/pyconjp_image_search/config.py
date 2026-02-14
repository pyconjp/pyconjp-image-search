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

# Embedding – SigLIP
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
EMBEDDING_DIM = 768
EMBEDDING_IMAGE_SIZE = 224

# Embedding – CLIP-L
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_DB_PATH = PROJECT_ROOT / "pyconjp_image_search_clip.duckdb"
CLIP_EMBEDDING_DIM = 768
