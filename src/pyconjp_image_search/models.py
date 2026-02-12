"""Data models for image metadata."""

from dataclasses import dataclass
from datetime import datetime


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
