"""CRUD operations for object detections in DuckDB."""

import duckdb

from pyconjp_image_search.models import ObjectDetection


def insert_object_detections(
    conn: duckdb.DuckDBPyConnection,
    detections: list[ObjectDetection],
) -> None:
    """Batch insert object detection results."""
    for det in detections:
        conn.execute(
            """
            INSERT INTO object_detections
            (detection_id, image_id, model_name, label, confidence,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (detection_id) DO NOTHING
            """,
            [
                det.detection_id,
                det.image_id,
                det.model_name,
                det.label,
                det.confidence,
                det.bbox_x1,
                det.bbox_y1,
                det.bbox_x2,
                det.bbox_y2,
            ],
        )


def mark_image_processed(
    conn: duckdb.DuckDBPyConnection,
    image_id: int,
    model_name: str,
    object_count: int,
) -> None:
    """Record that an image has been processed for object detection."""
    conn.execute(
        """
        INSERT INTO object_processed_images (image_id, model_name, object_count)
        VALUES (?, ?, ?)
        ON CONFLICT (image_id, model_name) DO UPDATE SET
            object_count = EXCLUDED.object_count,
            processed_at = now()
        """,
        [image_id, model_name, object_count],
    )


def get_object_processed_image_ids(
    conn: duckdb.DuckDBPyConnection,
    model_name: str,
) -> set[int]:
    """Return image IDs that have been processed (including those with zero detections)."""
    rows = conn.execute(
        "SELECT image_id FROM object_processed_images WHERE model_name = ?",
        [model_name],
    ).fetchall()
    return {row[0] for row in rows}


def get_object_stats(
    conn: duckdb.DuckDBPyConnection,
    model_name: str,
) -> tuple[int, int, int]:
    """Return (total_images, processed_images, detected_objects) for the model."""
    total_row = conn.execute(
        "SELECT COUNT(*) FROM images WHERE relative_path IS NOT NULL"
    ).fetchone()
    total = total_row[0] if total_row else 0

    processed_row = conn.execute(
        "SELECT COUNT(*) FROM object_processed_images WHERE model_name = ?",
        [model_name],
    ).fetchone()
    processed = processed_row[0] if processed_row else 0

    objects_row = conn.execute(
        "SELECT COUNT(*) FROM object_detections WHERE model_name = ?",
        [model_name],
    ).fetchone()
    objects = objects_row[0] if objects_row else 0

    return total, processed, objects


def get_objects_for_image(
    conn: duckdb.DuckDBPyConnection,
    image_id: int,
    model_name: str,
) -> list[ObjectDetection]:
    """Return all object detections for a given image."""
    rows = conn.execute(
        """
        SELECT detection_id, image_id, model_name, label, confidence,
               bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM object_detections
        WHERE image_id = ? AND model_name = ?
        ORDER BY confidence DESC
        """,
        [image_id, model_name],
    ).fetchall()
    return [_row_to_object_detection(row) for row in rows]


def _row_to_object_detection(row: tuple) -> ObjectDetection:
    """Convert a database row to an ObjectDetection object."""
    return ObjectDetection(
        detection_id=row[0],
        image_id=row[1],
        model_name=row[2],
        label=row[3],
        confidence=row[4],
        bbox_x1=row[5],
        bbox_y1=row[6],
        bbox_x2=row[7],
        bbox_y2=row[8],
    )
