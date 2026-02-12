"""Tests for search query functions."""

import numpy as np
from conftest import make_metadata

from pyconjp_image_search.embedding.repository import insert_embeddings
from pyconjp_image_search.manager.repository import insert_image
from pyconjp_image_search.search.query import (
    get_event_names,
    get_image_embedding,
    search_images_by_text,
)


def test_get_event_names(db_conn):
    insert_image(db_conn, make_metadata("1", event_name="PyCon JP"))
    insert_image(db_conn, make_metadata("2", event_name="PyCon US"))
    insert_image(db_conn, make_metadata("3", event_name="PyCon JP"))
    names = get_event_names(db_conn)
    assert names == ["PyCon JP", "PyCon US"]


def test_get_image_embedding(db_conn):
    insert_image(db_conn, make_metadata("1"))
    image_ids = [row[0] for row in db_conn.execute("SELECT id FROM images ORDER BY id").fetchall()]

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((1, 768)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    model = "test-model"
    insert_embeddings(db_conn, image_ids, embeddings, model)

    # Existing embedding
    result = get_image_embedding(db_conn, image_ids[0], model)
    assert result is not None
    assert result.shape == (768,)
    np.testing.assert_allclose(result, embeddings[0], atol=1e-6)

    # Non-existent image
    assert get_image_embedding(db_conn, 9999, model) is None

    # Non-existent model
    assert get_image_embedding(db_conn, image_ids[0], "no-model") is None


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


def test_search_images_by_text_with_event_filter(db_conn):
    insert_image(db_conn, make_metadata("1", event_name="PyCon JP 2024"))
    insert_image(db_conn, make_metadata("2", event_name="PyCon JP 2023"))
    image_ids = [row[0] for row in db_conn.execute("SELECT id FROM images ORDER BY id").fetchall()]

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((2, 768)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    model = "test-model"
    insert_embeddings(db_conn, image_ids, embeddings, model)

    # Filter to only "PyCon JP 2024"
    results = search_images_by_text(
        db_conn, embeddings[0:1], model, limit=10, event_names=["PyCon JP 2024"],
    )
    assert len(results) == 1
    assert results[0][0].event_name == "PyCon JP 2024"

    # No filter returns all
    results_all = search_images_by_text(db_conn, embeddings[0:1], model, limit=10)
    assert len(results_all) == 2


def test_search_images_by_text_offset(db_conn):
    for i in range(3):
        insert_image(db_conn, make_metadata(str(i + 1)))
    image_ids = [row[0] for row in db_conn.execute("SELECT id FROM images ORDER BY id").fetchall()]

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((3, 768)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    model = "test-model"
    insert_embeddings(db_conn, image_ids, embeddings, model)

    # First page
    page1 = search_images_by_text(db_conn, embeddings[0:1], model, limit=2, offset=0)
    assert len(page1) == 2

    # Second page
    page2 = search_images_by_text(db_conn, embeddings[0:1], model, limit=2, offset=2)
    assert len(page2) == 1

    # No overlap
    page1_ids = {m.id for m, _ in page1}
    page2_ids = {m.id for m, _ in page2}
    assert page1_ids.isdisjoint(page2_ids)
