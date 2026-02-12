"""CRUD operations for image metadata in DuckDB."""

import duckdb

from pyconjp_image_search.models import ImageMetadata


def insert_image(conn: duckdb.DuckDBPyConnection, meta: ImageMetadata) -> None:
    """Insert a single image metadata record. Skip on URL conflict."""
    conn.execute(
        """
        INSERT INTO images (
            image_url, relative_path, local_filename,
            flickr_photo_id, album_id, album_title,
            event_name, event_year, event_type,
            image_format, width, height, file_size_bytes, downloaded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (image_url) DO NOTHING
        """,
        [
            meta.image_url,
            meta.relative_path,
            meta.local_filename,
            meta.flickr_photo_id,
            meta.album_id,
            meta.album_title,
            meta.event_name,
            meta.event_year,
            meta.event_type,
            meta.image_format,
            meta.width,
            meta.height,
            meta.file_size_bytes,
            meta.downloaded_at,
        ],
    )


def insert_images(conn: duckdb.DuckDBPyConnection, metadata_list: list[ImageMetadata]) -> None:
    """Bulk insert image metadata records."""
    for meta in metadata_list:
        insert_image(conn, meta)


def get_existing_photo_ids(
    conn: duckdb.DuckDBPyConnection, album_id: str | None = None
) -> set[str]:
    """Return set of flickr_photo_id values already in DB."""
    query = "SELECT flickr_photo_id FROM images WHERE flickr_photo_id IS NOT NULL"
    params: list = []
    if album_id:
        query += " AND album_id = ?"
        params.append(album_id)
    rows = conn.execute(query, params).fetchall()
    return {row[0] for row in rows}


def get_image_by_url(conn: duckdb.DuckDBPyConnection, url: str) -> ImageMetadata | None:
    """Look up a single image by URL."""
    result = conn.execute("SELECT * FROM images WHERE image_url = ?", [url]).fetchone()
    if result is None:
        return None
    return _row_to_metadata(result)


def list_images(
    conn: duckdb.DuckDBPyConnection,
    event_name: str | None = None,
    event_year: int | None = None,
    album_id: str | None = None,
) -> list[ImageMetadata]:
    """List images with optional filters."""
    query = "SELECT * FROM images WHERE 1=1"
    params: list = []
    if event_name is not None:
        query += " AND event_name = ?"
        params.append(event_name)
    if event_year is not None:
        query += " AND event_year = ?"
        params.append(event_year)
    if album_id is not None:
        query += " AND album_id = ?"
        params.append(album_id)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    return [_row_to_metadata(row) for row in rows]


def _row_to_metadata(row: tuple) -> ImageMetadata:
    """Convert a DB row tuple to ImageMetadata.

    Column order matches schema.py DDL:
    0:id, 1:image_url, 2:relative_path, 3:local_filename,
    4:flickr_photo_id, 5:album_id, 6:album_title,
    7:event_name, 8:event_year, 9:event_type,
    10:image_format, 11:width, 12:height, 13:file_size_bytes,
    14:downloaded_at, 15:created_at
    """
    return ImageMetadata(
        id=row[0],
        image_url=row[1],
        relative_path=row[2],
        local_filename=row[3],
        flickr_photo_id=row[4],
        album_id=row[5],
        album_title=row[6],
        event_name=row[7],
        event_year=row[8],
        event_type=row[9],
        image_format=row[10],
        width=row[11],
        height=row[12],
        file_size_bytes=row[13],
        downloaded_at=row[14],
        created_at=row[15],
    )
