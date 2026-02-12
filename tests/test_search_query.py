"""Tests for search query functions."""

import numpy as np
from conftest import make_metadata

from pyconjp_image_search.embedding.repository import insert_embeddings
from pyconjp_image_search.manager.repository import insert_image
from pyconjp_image_search.search.query import (
    get_event_names,
    get_event_years,
    search_images,
    search_images_by_text,
)


def test_search_images_no_filter(db_conn):
    insert_image(db_conn, make_metadata("1"))
    insert_image(db_conn, make_metadata("2"))
    results = search_images(db_conn)
    assert len(results) == 2


def test_search_images_by_event_name(db_conn):
    insert_image(db_conn, make_metadata("1", event_name="PyCon JP"))
    insert_image(db_conn, make_metadata("2", event_name="PyCon US"))
    results = search_images(db_conn, event_name="PyCon JP")
    assert len(results) == 1
    assert results[0].event_name == "PyCon JP"


def test_search_images_by_year(db_conn):
    insert_image(db_conn, make_metadata("1", event_year=2024))
    insert_image(db_conn, make_metadata("2", event_year=2025))
    results = search_images(db_conn, event_year=2024)
    assert len(results) == 1
    assert results[0].event_year == 2024


def test_get_event_names(db_conn):
    insert_image(db_conn, make_metadata("1", event_name="PyCon JP"))
    insert_image(db_conn, make_metadata("2", event_name="PyCon US"))
    insert_image(db_conn, make_metadata("3", event_name="PyCon JP"))
    names = get_event_names(db_conn)
    assert names == ["PyCon JP", "PyCon US"]


def test_get_event_years(db_conn):
    insert_image(db_conn, make_metadata("1", event_year=2023))
    insert_image(db_conn, make_metadata("2", event_year=2025))
    insert_image(db_conn, make_metadata("3", event_year=2024))
    years = get_event_years(db_conn)
    assert years == [2025, 2024, 2023]  # DESC order


def test_search_images_by_text_cosine(db_conn):
    # Insert images
    insert_image(db_conn, make_metadata("1"))
    insert_image(db_conn, make_metadata("2"))
    image_ids = [row[0] for row in db_conn.execute("SELECT id FROM images ORDER BY id").fetchall()]

    # Insert embeddings
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((2, 768)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    model = "test-model"
    insert_embeddings(db_conn, image_ids, embeddings, model)

    # Search with first embedding
    results = search_images_by_text(db_conn, embeddings[0:1], model, limit=2)
    assert len(results) == 2
    top_meta, top_score = results[0]
    assert top_meta.id == image_ids[0]
    assert top_score > 0.99
