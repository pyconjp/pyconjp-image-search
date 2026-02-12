"""Shared test fixtures."""

from datetime import UTC, datetime

import duckdb
import pytest

from pyconjp_image_search.manager.schema import ensure_schema
from pyconjp_image_search.models import ImageMetadata


@pytest.fixture
def db_conn():
    """In-memory DuckDB connection with schema initialized."""
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def sample_metadata() -> ImageMetadata:
    """A single ImageMetadata fixture."""
    return ImageMetadata(
        id=None,
        image_url="https://farm1.staticflickr.com/server/12345_secret_b.jpg",
        relative_path="test_album/12345.jpg",
        local_filename="12345.jpg",
        flickr_photo_id="12345",
        album_id="album_001",
        album_title="Test Album",
        event_name="PyCon JP",
        event_year=2024,
        event_type="conference",
        image_format="JPEG",
        width=1024,
        height=768,
        file_size_bytes=102400,
        downloaded_at=datetime(2024, 1, 1, tzinfo=UTC),
        created_at=None,
    )


def make_metadata(
    photo_id: str,
    event_name: str = "PyCon JP",
    event_year: int = 2024,
    album_id: str = "album_001",
) -> ImageMetadata:
    """Helper to create ImageMetadata with unique fields."""
    return ImageMetadata(
        id=None,
        image_url=f"https://example.com/{photo_id}.jpg",
        relative_path=f"album/{photo_id}.jpg",
        local_filename=f"{photo_id}.jpg",
        flickr_photo_id=photo_id,
        album_id=album_id,
        album_title="Test Album",
        event_name=event_name,
        event_year=event_year,
        event_type="conference",
        image_format="JPEG",
        width=1024,
        height=768,
        file_size_bytes=50000,
        downloaded_at=datetime(2024, 1, 1, tzinfo=UTC),
        created_at=None,
    )
