"""Shared DuckDB connection factory."""

import duckdb

from pyconjp_image_search.config import DB_PATH


def get_connection(db_path: str | None = None, embedding_dim: int = 768) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection. Defaults to the project-root DB file."""
    path = db_path or str(DB_PATH)
    conn = duckdb.connect(path)

    from pyconjp_image_search.manager.schema import ensure_schema

    ensure_schema(conn, embedding_dim=embedding_dim)
    return conn
