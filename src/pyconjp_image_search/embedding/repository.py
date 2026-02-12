"""CRUD operations for image embeddings in DuckDB."""

import duckdb
import numpy as np

from pyconjp_image_search.manager.repository import _row_to_metadata
from pyconjp_image_search.models import ImageMetadata


def insert_embeddings(
    conn: duckdb.DuckDBPyConnection,
    image_ids: list[int],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    """Batch insert embeddings. Skips on conflict (idempotent)."""
    for i, image_id in enumerate(image_ids):
        vec = embeddings[i].tolist()
        conn.execute(
            """
            INSERT INTO image_embeddings (image_id, model_name, embedding)
            VALUES (?, ?, ?)
            ON CONFLICT (image_id, model_name) DO NOTHING
            """,
            [image_id, model_name, vec],
        )


def get_unembedded_image_ids(
    conn: duckdb.DuckDBPyConnection,
    model_name: str,
) -> list[tuple[int, str]]:
    """Return (image_id, relative_path) pairs for images without embeddings."""
    rows = conn.execute(
        """
        SELECT i.id, i.relative_path
        FROM images i
        LEFT JOIN image_embeddings e
            ON i.id = e.image_id AND e.model_name = ?
        WHERE e.image_id IS NULL
            AND i.relative_path IS NOT NULL
        ORDER BY i.id
        """,
        [model_name],
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def get_embedding_stats(
    conn: duckdb.DuckDBPyConnection,
    model_name: str,
) -> tuple[int, int]:
    """Return (total_images, embedded_count) for a given model."""
    total_row = conn.execute(
        "SELECT COUNT(*) FROM images WHERE relative_path IS NOT NULL"
    ).fetchone()
    total = total_row[0] if total_row else 0
    embedded_row = conn.execute(
        "SELECT COUNT(*) FROM image_embeddings WHERE model_name = ?",
        [model_name],
    ).fetchone()
    embedded = embedded_row[0] if embedded_row else 0
    return total, embedded


def search_by_embedding(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: np.ndarray,
    model_name: str,
    limit: int = 20,
) -> list[tuple[ImageMetadata, float]]:
    """Search images by cosine similarity to query embedding."""
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
