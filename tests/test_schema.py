"""Tests for DuckDB schema creation and migration."""

import duckdb

from pyconjp_image_search.manager.schema import ensure_schema


def test_ensure_schema_creates_images_table(db_conn):
    tables = db_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'images'"
    ).fetchall()
    assert len(tables) == 1


def test_ensure_schema_creates_embeddings_table(db_conn):
    tables = db_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'image_embeddings'"
    ).fetchall()
    assert len(tables) == 1


def test_ensure_schema_idempotent():
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    ensure_schema(conn)  # Should not raise
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    table_names = {row[0] for row in tables}
    assert "images" in table_names
    assert "image_embeddings" in table_names
    conn.close()
