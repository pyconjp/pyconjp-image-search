"""DuckDB schema definition and migration."""

import duckdb


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables and indexes if they do not exist."""
    conn.execute("CREATE SEQUENCE IF NOT EXISTS images_id_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id              INTEGER PRIMARY KEY DEFAULT nextval('images_id_seq'),
            image_url       VARCHAR NOT NULL UNIQUE,
            relative_path   VARCHAR,
            local_filename  VARCHAR,
            flickr_photo_id VARCHAR UNIQUE,
            album_id        VARCHAR,
            album_title     VARCHAR,
            event_name      VARCHAR NOT NULL,
            event_year      INTEGER NOT NULL,
            event_type      VARCHAR DEFAULT 'conference',
            image_format    VARCHAR,
            width           INTEGER,
            height          INTEGER,
            file_size_bytes BIGINT,
            downloaded_at   TIMESTAMP,
            created_at      TIMESTAMP DEFAULT current_timestamp
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_event ON images(event_name, event_year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_album ON images(album_id)")

    # image_embeddings table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_embeddings (
            image_id    INTEGER NOT NULL,
            model_name  VARCHAR NOT NULL,
            embedding   FLOAT[768],
            created_at  TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (image_id, model_name),
            FOREIGN KEY (image_id) REFERENCES images(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON image_embeddings(model_name)")

    # Migration for existing databases
    _migrate(conn)


def _migrate(conn: duckdb.DuckDBPyConnection) -> None:
    """Add columns that may not exist in older schemas."""
    migrations = [
        "ALTER TABLE images ADD COLUMN IF NOT EXISTS relative_path VARCHAR",
        "ALTER TABLE images ADD COLUMN IF NOT EXISTS flickr_photo_id VARCHAR",
        "ALTER TABLE images ADD COLUMN IF NOT EXISTS album_id VARCHAR",
        "ALTER TABLE images ADD COLUMN IF NOT EXISTS album_title VARCHAR",
    ]
    for sql in migrations:
        conn.execute(sql)
    # Add unique index on flickr_photo_id (separate from column creation)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_images_flickr_photo_id ON images(flickr_photo_id)"
    )
