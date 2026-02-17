"""CRUD operations for face detections in DuckDB."""

import json

import duckdb
import numpy as np

from pyconjp_image_search.manager.repository import _row_to_metadata
from pyconjp_image_search.models import FaceDetection, ImageMetadata


def insert_face_detections(
    conn: duckdb.DuckDBPyConnection,
    detections: list[FaceDetection],
) -> None:
    """Batch insert face detection results."""
    for det in detections:
        landmark_json = json.dumps(det.landmark) if det.landmark is not None else None
        conn.execute(
            """
            INSERT INTO face_detections
            (face_id, image_id, model_name, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             det_score, landmark, age, gender, embedding, person_label, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (face_id) DO NOTHING
            """,
            [
                det.face_id,
                det.image_id,
                det.model_name,
                det.bbox[0],
                det.bbox[1],
                det.bbox[2],
                det.bbox[3],
                det.det_score,
                landmark_json,
                det.age,
                det.gender,
                det.embedding.tolist(),
                det.person_label,
                det.cluster_id,
            ],
        )


def mark_image_processed(
    conn: duckdb.DuckDBPyConnection,
    image_id: int,
    model_name: str,
    face_count: int,
) -> None:
    """Record that an image has been processed for face detection."""
    conn.execute(
        """
        INSERT INTO face_processed_images (image_id, model_name, face_count)
        VALUES (?, ?, ?)
        ON CONFLICT (image_id, model_name) DO UPDATE SET
            face_count = EXCLUDED.face_count,
            processed_at = current_timestamp
        """,
        [image_id, model_name, face_count],
    )


def get_face_processed_image_ids(
    conn: duckdb.DuckDBPyConnection,
    model_name: str,
) -> set[int]:
    """Return image IDs that have been processed (including those with zero faces)."""
    rows = conn.execute(
        "SELECT image_id FROM face_processed_images WHERE model_name = ?",
        [model_name],
    ).fetchall()
    return {row[0] for row in rows}


def get_face_stats(
    conn: duckdb.DuckDBPyConnection,
    model_name: str,
) -> tuple[int, int, int]:
    """Return (total_images, processed_images, detected_faces) for the model."""
    total_row = conn.execute(
        "SELECT COUNT(*) FROM images WHERE relative_path IS NOT NULL"
    ).fetchone()
    total = total_row[0] if total_row else 0

    processed_row = conn.execute(
        "SELECT COUNT(*) FROM face_processed_images WHERE model_name = ?",
        [model_name],
    ).fetchone()
    processed = processed_row[0] if processed_row else 0

    faces_row = conn.execute(
        "SELECT COUNT(*) FROM face_detections WHERE model_name = ?",
        [model_name],
    ).fetchone()
    faces = faces_row[0] if faces_row else 0

    return total, processed, faces


def search_faces_by_embedding(
    conn: duckdb.DuckDBPyConnection,
    query_embedding: np.ndarray,
    model_name: str,
    limit: int = 20,
    event_names: list[str] | None = None,
) -> list[tuple[FaceDetection, ImageMetadata, float]]:
    """Search faces by cosine similarity to a query embedding."""
    query_vec = query_embedding.flatten().tolist()
    params: list = [query_vec, model_name]

    where_clauses = ["f.model_name = ?"]
    if event_names:
        placeholders = ", ".join(["?"] * len(event_names))
        where_clauses.append(f"i.event_name IN ({placeholders})")
        params.extend(event_names)

    where_sql = " AND ".join(where_clauses)
    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT f.face_id, f.image_id, f.model_name,
               f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
               f.det_score, f.landmark, f.age, f.gender,
               f.embedding, f.person_label, f.cluster_id,
               i.*,
               list_cosine_similarity(f.embedding, ?::FLOAT[512]) AS score
        FROM face_detections f
        JOIN images i ON i.id = f.image_id
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    results = []
    for row in rows:
        face = _row_to_face_detection(row[:14])
        score = row[-1]
        meta = _row_to_metadata(row[14:-1])
        results.append((face, meta, score))
    return results


def get_faces_for_image(
    conn: duckdb.DuckDBPyConnection,
    image_id: int,
    model_name: str,
) -> list[FaceDetection]:
    """Return all face detections for a given image."""
    rows = conn.execute(
        """
        SELECT face_id, image_id, model_name,
               bbox_x1, bbox_y1, bbox_x2, bbox_y2,
               det_score, landmark, age, gender,
               embedding, person_label, cluster_id
        FROM face_detections
        WHERE image_id = ? AND model_name = ?
        ORDER BY det_score DESC
        """,
        [image_id, model_name],
    ).fetchall()
    return [_row_to_face_detection(row) for row in rows]


def _row_to_face_detection(row: tuple) -> FaceDetection:
    """Convert a database row to a FaceDetection object."""
    landmark_raw = row[8]
    if isinstance(landmark_raw, str):
        landmark = json.loads(landmark_raw)
    else:
        landmark = landmark_raw

    embedding_raw = row[11]
    if embedding_raw is not None:
        embedding = np.array(embedding_raw, dtype=np.float32)
    else:
        embedding = np.zeros(512, dtype=np.float32)

    return FaceDetection(
        face_id=row[0],
        image_id=row[1],
        model_name=row[2],
        bbox=(row[3], row[4], row[5], row[6]),
        det_score=row[7],
        landmark=landmark,
        age=row[9],
        gender=row[10],
        embedding=embedding,
        person_label=row[12],
        cluster_id=row[13],
    )
