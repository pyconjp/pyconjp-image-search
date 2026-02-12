"""Tests for ImageMetadata dataclass."""

from datetime import UTC, datetime

from pyconjp_image_search.models import ImageMetadata


def test_image_metadata_creation():
    meta = ImageMetadata(
        id=1,
        image_url="https://example.com/photo.jpg",
        relative_path="album/photo.jpg",
        local_filename="photo.jpg",
        flickr_photo_id="12345",
        album_id="album_001",
        album_title="Test",
        event_name="PyCon JP",
        event_year=2024,
        event_type="conference",
        image_format="JPEG",
        width=1024,
        height=768,
        file_size_bytes=50000,
        downloaded_at=datetime(2024, 1, 1, tzinfo=UTC),
        created_at=None,
    )
    assert meta.event_name == "PyCon JP"
    assert meta.event_year == 2024
    assert meta.width == 1024
    assert meta.created_at is None


def test_image_metadata_optional_fields():
    meta = ImageMetadata(
        id=None,
        image_url="https://example.com/photo.jpg",
        relative_path=None,
        local_filename=None,
        flickr_photo_id=None,
        album_id=None,
        album_title=None,
        event_name="PyCon JP",
        event_year=2024,
        event_type="conference",
        image_format=None,
        width=None,
        height=None,
        file_size_bytes=None,
        downloaded_at=None,
        created_at=None,
    )
    assert meta.id is None
    assert meta.relative_path is None
    assert meta.flickr_photo_id is None
