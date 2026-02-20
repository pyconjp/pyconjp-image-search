"""Data models for image metadata."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class ImageMetadata:
    """Metadata for a single image."""

    id: int | None
    image_url: str
    relative_path: str | None
    local_filename: str | None
    flickr_photo_id: str | None
    album_id: str | None
    album_title: str | None
    event_name: str
    event_year: int
    event_type: str
    image_format: str | None
    width: int | None
    height: int | None
    file_size_bytes: int | None
    downloaded_at: datetime | None
    created_at: datetime | None


@dataclass
class ObjectDetection:
    """A single detected object within an image."""

    detection_id: str
    image_id: int
    model_name: str
    label: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float


@dataclass
class FaceDetection:
    """A single detected face within an image."""

    face_id: str
    image_id: int
    model_name: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    det_score: float
    landmark: list | None
    age: int | None
    gender: str | None  # "M" or "F"
    embedding: np.ndarray  # shape (512,)
    person_label: str | None
    cluster_id: int | None
