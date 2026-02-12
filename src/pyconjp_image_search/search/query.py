"""Search queries against DuckDB."""

import duckdb
import numpy as np

from pyconjp_image_search.manager.repository import _row_to_metadata
from pyconjp_image_search.models import ImageMetadata


def search_images(
    conn: duckdb.DuckDBPyConnection,
    event_name: str | None = None,
    event_year: int | None = None,
    limit: int = 50,
) -> list[ImageMetadata]:
    """Search images with optional filters."""
    query = "SELECT * FROM images WHERE 1=1"
    params: list = []

    if event_name:
        query += " AND event_name = ?"
        params.append(event_name)
    if event_year:
        query += " AND event_year = ?"
        params.append(event_year)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [_row_to_metadata(row) for row in rows]


def get_event_names(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Get distinct event names from the database."""
    rows = conn.execute("SELECT DISTINCT event_name FROM images ORDER BY event_name").fetchall()
    return [row[0] for row in rows]


def get_event_years(conn: duckdb.DuckDBPyConnection) -> list[int]:
    """Get distinct event years from the database."""
    rows = conn.execute(
        "SELECT DISTINCT event_year FROM images ORDER BY event_year DESC"
    ).fetchall()
    return [row[0] for row in rows]


def search_images_by_text(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: np.ndarray,
    model_name: str,
    limit: int = 20,
) -> list[tuple[ImageMetadata, float]]:
    """Search images by cosine similarity to a text embedding."""
    query_vec = query_embedding.flatten().tolist()
    rows = conn.execute(
        """
        SELECT i.*, list_cosine_similarity(e.embedding, ?::FLOAT[768]) AS score
        FROM image_embeddings e
        JOIN images i ON i.id = e.image_id
        WHERE e.model_name = ?
        ORDER BY score DESC
        LIMIT ?
        """,
        [query_vec, model_name, limit],
    ).fetchall()
    results = []
    for row in rows:
        score = row[-1]
        meta = _row_to_metadata(row[:-1])
        results.append((meta, score))
    return results
