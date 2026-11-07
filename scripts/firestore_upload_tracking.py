"""Track which documents have been uploaded to Firestore, in DuckDB.

upload_to_firestore.py used to stream every existing document ID from
Firestore to decide what to upload, which costs one read per document
(~100k reads) on every run. Recording uploaded IDs locally lets the diff be
computed in DuckDB with zero Firestore reads.

This module depends only on duckdb so it can be unit-tested without the
Firestore client.
"""

from collections.abc import Iterable

import duckdb

TRACKING_TABLE = "firestore_uploads"


def ensure_tracking_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the tracking table if it does not exist."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TRACKING_TABLE} (
            collection  VARCHAR NOT NULL,
            doc_id      VARCHAR NOT NULL,
            uploaded_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (collection, doc_id)
        )
    """)


def tracking_table_exists(conn: duckdb.DuckDBPyConnection) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = ?", [TRACKING_TABLE]
    ).fetchone()
    return bool(row and row[0])


def get_uploaded_ids(conn: duckdb.DuckDBPyConnection, collection: str) -> set[str]:
    """Return doc IDs recorded as uploaded for a collection (empty if no table)."""
    if not tracking_table_exists(conn):
        return set()
    rows = conn.execute(
        f"SELECT doc_id FROM {TRACKING_TABLE} WHERE collection = ?", [collection]
    ).fetchall()
    return {row[0] for row in rows}


def record_uploaded(
    conn: duckdb.DuckDBPyConnection, collection: str, doc_ids: Iterable[str]
) -> int:
    """Record doc IDs as uploaded. Already-recorded IDs are ignored.

    Returns the number of IDs passed in.
    """
    ids = list(doc_ids)
    if not ids:
        return 0
    conn.execute(
        f"""
        INSERT INTO {TRACKING_TABLE} (collection, doc_id)
        SELECT ?, unnest(?::VARCHAR[])
        ON CONFLICT DO NOTHING
        """,
        [collection, ids],
    )
    return len(ids)
