"""Tests for embedding repository CRUD operations."""

import numpy as np
from conftest import make_metadata

from pyconjp_image_search.embedding.repository import (
    get_embedding_stats,
    get_unembedded_image_ids,
    insert_embeddings,
    search_by_embedding,
)
from pyconjp_image_search.manager.repository import insert_image

MODEL = "test-model"


def _insert_test_images(db_conn, count=3):
    """Insert test images and return their IDs."""
    for i in range(count):
        meta = make_metadata(str(1000 + i))
        insert_image(db_conn, meta)
    rows = db_conn.execute("SELECT id FROM images ORDER BY id").fetchall()
    return [row[0] for row in rows]


def test_insert_and_search_embeddings(db_conn):
    image_ids = _insert_test_images(db_conn, 3)

    # Create normalized embeddings
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((3, 768)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    insert_embeddings(db_conn, image_ids, embeddings, MODEL)

    # Search with the first embedding — should return itself as top result
    results = search_by_embedding(db_conn, embeddings[0:1], MODEL, limit=3)
    assert len(results) == 3
    top_meta, top_score = results[0]
    assert top_meta.id == image_ids[0]
    assert top_score > 0.99  # Cosine similarity with itself


def test_insert_embeddings_idempotent(db_conn):
    image_ids = _insert_test_images(db_conn, 1)
    emb = np.ones((1, 768), dtype=np.float32)
    emb = emb / np.linalg.norm(emb)

    insert_embeddings(db_conn, image_ids, emb, MODEL)
    insert_embeddings(db_conn, image_ids, emb, MODEL)  # Should not raise

    count = db_conn.execute("SELECT COUNT(*) FROM image_embeddings").fetchone()[0]
    assert count == 1


def test_get_unembedded_image_ids(db_conn):
    image_ids = _insert_test_images(db_conn, 3)

    # Embed only the first image
    emb = np.ones((1, 768), dtype=np.float32)
    insert_embeddings(db_conn, [image_ids[0]], emb, MODEL)

    unembedded = get_unembedded_image_ids(db_conn, MODEL)
    unembedded_ids = [item[0] for item in unembedded]
    assert image_ids[0] not in unembedded_ids
    assert len(unembedded) == 2


def test_get_embedding_stats(db_conn):
    image_ids = _insert_test_images(db_conn, 5)

    total, embedded = get_embedding_stats(db_conn, MODEL)
    assert total == 5
    assert embedded == 0

    emb = np.ones((2, 768), dtype=np.float32)
    insert_embeddings(db_conn, image_ids[:2], emb, MODEL)

    total, embedded = get_embedding_stats(db_conn, MODEL)
    assert total == 5
    assert embedded == 2
