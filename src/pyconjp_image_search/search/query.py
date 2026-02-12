"""Search queries against DuckDB."""

import duckdb
import numpy as np

from pyconjp_image_search.manager.repository import _row_to_metadata
from pyconjp_image_search.models import ImageMetadata


def get_event_names(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Get distinct event names from the database."""
    rows = conn.execute("SELECT DISTINCT event_name FROM images ORDER BY event_name").fetchall()
    return [row[0] for row in rows]


def search_images_by_text(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: np.ndarray,
    model_name: str,
    limit: int = 20,
    offset: int = 0,
    event_names: list[str] | None = None,
) -> list[tuple[ImageMetadata, float]]:
    """Search images by cosine similarity to a text embedding."""
    query_vec = query_embedding.flatten().tolist()
    params: list = [query_vec, model_name]

    where_clauses = ["e.model_name = ?"]
    if event_names:
        placeholders = ", ".join(["?"] * len(event_names))
        where_clauses.append(f"i.event_name IN ({placeholders})")
        params.extend(event_names)

    where_sql = " AND ".join(where_clauses)
    params.extend([limit, offset])

    rows = conn.execute(
        f"""
        SELECT i.*, list_cosine_similarity(e.embedding, ?::FLOAT[768]) AS score
        FROM image_embeddings e
        JOIN images i ON i.id = e.image_id
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT ?
        OFFSET ?
        """,
        params,
    ).fetchall()
    results = []
    for row in rows:
        score = row[-1]
        meta = _row_to_metadata(row[:-1])
        results.append((meta, score))
    return results
