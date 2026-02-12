"""Tests for images table CRUD operations."""

from conftest import make_metadata

from pyconjp_image_search.manager.repository import (
    get_existing_photo_ids,
    get_image_by_url,
    insert_image,
    list_images,
)


def test_insert_and_get_by_url(db_conn, sample_metadata):
    insert_image(db_conn, sample_metadata)
    result = get_image_by_url(db_conn, sample_metadata.image_url)
    assert result is not None
    assert result.event_name == "PyCon JP"
    assert result.event_year == 2024
    assert result.flickr_photo_id == "12345"


def test_insert_duplicate_ignored(db_conn, sample_metadata):
    insert_image(db_conn, sample_metadata)
    insert_image(db_conn, sample_metadata)  # Should not raise
    count = db_conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    assert count == 1


def test_list_images_filter_by_event(db_conn):
    insert_image(db_conn, make_metadata("1", event_name="PyCon JP", event_year=2024))
    insert_image(db_conn, make_metadata("2", event_name="PyCon JP", event_year=2025))
    insert_image(db_conn, make_metadata("3", event_name="PyCon US", event_year=2024))

    results = list_images(db_conn, event_name="PyCon JP")
    assert len(results) == 2

    results = list_images(db_conn, event_year=2024)
    assert len(results) == 2

    results = list_images(db_conn, event_name="PyCon JP", event_year=2024)
    assert len(results) == 1


def test_get_existing_photo_ids(db_conn):
    insert_image(db_conn, make_metadata("100", album_id="a1"))
    insert_image(db_conn, make_metadata("200", album_id="a1"))
    insert_image(db_conn, make_metadata("300", album_id="a2"))

    ids = get_existing_photo_ids(db_conn, album_id="a1")
    assert ids == {"100", "200"}

    all_ids = get_existing_photo_ids(db_conn)
    assert all_ids == {"100", "200", "300"}
