# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-cloud-firestore>=2.19",
#     "duckdb>=1.0",
# ]
# ///
"""Upload DuckDB data to Firestore.

Setup (認証):
    # gcloud CLI のインストール: https://cloud.google.com/sdk/docs/install
    # Application Default Credentials でログイン
    gcloud auth application-default login --project pyconjp-image-search

Usage:
    cd /path/to/pyconjp-image-search
    uv run scripts/upload_to_firestore.py \
        [--db pyconjp_image_search.duckdb] \
        [--project pyconjp-image-search] [--dry-run]
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb
from google.cloud.firestore_v1 import Client
from google.cloud.firestore_v1.vector import Vector

SIGLIP_MODEL = "google/siglip2-base-patch16-224"
BATCH_LIMIT = 500  # Firestore batch write limit
SLEEP_SECONDS = 1  # Sleep between batch commits to avoid quota exceeded


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload DuckDB data to Firestore")
    parser.add_argument(
        "--db",
        default="pyconjp_image_search.duckdb",
        help="Path to DuckDB file (default: pyconjp_image_search.duckdb)",
    )
    parser.add_argument(
        "--project",
        default="pyconjp-image-search",
        help="Firebase project ID (default: pyconjp-image-search)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without uploading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing documents instead of skipping them",
    )
    return parser.parse_args()


def _build_image_tags(conn: duckdb.DuckDBPyConnection) -> dict[int, list[str]]:
    """Build mapping of image_id -> list of unique object detection labels (tags)."""
    rows = conn.execute("""
        SELECT image_id, list(DISTINCT label ORDER BY label) AS tags
        FROM object_detections
        GROUP BY image_id
    """).fetchall()
    return {row[0]: row[1] for row in rows}


def _build_image_event_names(conn: duckdb.DuckDBPyConnection) -> dict[int, str]:
    """Build mapping of image_id -> event_name."""
    rows = conn.execute("SELECT id, event_name FROM images").fetchall()
    return {row[0]: row[1] for row in rows}


def upload_images(
    conn: duckdb.DuckDBPyConnection,
    fs_client: Client,
    dry_run: bool,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Upload images + embeddings. Returns mapping of DuckDB image_id -> flickr_photo_id."""
    rows = conn.execute(f"""
        SELECT
            i.id, i.image_url, i.flickr_photo_id, i.album_id, i.album_title,
            i.event_name, i.event_year, i.event_type, i.width, i.height,
            e.embedding
        FROM images i
        JOIN image_embeddings e ON e.image_id = i.id
        WHERE e.model_name = '{SIGLIP_MODEL}'
        ORDER BY i.id
    """).fetchall()

    columns = [
        "id",
        "image_url",
        "flickr_photo_id",
        "album_id",
        "album_title",
        "event_name",
        "event_year",
        "event_type",
        "width",
        "height",
        "embedding",
    ]

    # Build tags mapping for denormalization
    image_tags = _build_image_tags(conn)

    id_to_photo = {}
    print(f"Images: {len(rows)} documents to upload")

    if dry_run:
        for row in rows:
            data = dict(zip(columns, row))
            id_to_photo[data["id"]] = data["flickr_photo_id"]
        return id_to_photo

    # Get existing document IDs to skip (unless --force)
    existing_ids = set()
    if not force:
        for doc in fs_client.collection("images").select([]).stream():
            existing_ids.add(doc.id)
        if existing_ids:
            print(f"  Found {len(existing_ids)} existing images, skipping duplicates")
    else:
        print("  Force mode: overwriting existing documents")

    batch = fs_client.batch()
    batch_count = 0
    skipped = 0

    for i, row in enumerate(rows):
        data = dict(zip(columns, row))
        image_id = data.pop("id")
        flickr_photo_id = data["flickr_photo_id"]
        id_to_photo[image_id] = flickr_photo_id

        if flickr_photo_id in existing_ids:
            skipped += 1
            continue

        embedding = data.pop("embedding")
        data["embedding"] = Vector(list(embedding))

        # Denormalize tags from object_detections
        tags = image_tags.get(image_id, [])
        if tags:
            data["tags"] = tags

        doc_ref = fs_client.collection("images").document(flickr_photo_id)
        batch.set(doc_ref, data)
        batch_count += 1

        if batch_count >= BATCH_LIMIT:
            batch.commit()
            print(f"  Committed {i + 1 - skipped}/{len(rows) - skipped} images")
            time.sleep(SLEEP_SECONDS)
            batch = fs_client.batch()
            batch_count = 0

    if batch_count > 0:
        batch.commit()
    print(f"  Completed: {len(rows) - skipped} images uploaded ({skipped} already existed)")

    return id_to_photo


def upload_face_detections(
    conn: duckdb.DuckDBPyConnection,
    fs_client: Client,
    id_to_photo: dict[str, str],
    id_to_event: dict[int, str],
    dry_run: bool,
    *,
    force: bool = False,
) -> None:
    """Upload face detections with embeddings."""
    rows = conn.execute("""
        SELECT
            face_id, image_id, model_name,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            det_score, age, gender, embedding,
            person_label, cluster_id
        FROM face_detections
        ORDER BY image_id
    """).fetchall()

    columns = [
        "face_id",
        "image_id",
        "model_name",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "det_score",
        "age",
        "gender",
        "embedding",
        "person_label",
        "cluster_id",
    ]

    print(f"Face detections: {len(rows)} documents to upload")

    if dry_run:
        return

    # Get existing document IDs to skip (unless --force)
    existing_ids = set()
    if not force:
        for doc in fs_client.collection("face_detections").select([]).stream():
            existing_ids.add(doc.id)
        if existing_ids:
            print(f"  Found {len(existing_ids)} existing face detections, skipping duplicates")
    else:
        print("  Force mode: overwriting existing documents")

    batch = fs_client.batch()
    batch_count = 0
    skipped = 0

    for i, row in enumerate(rows):
        data = dict(zip(columns, row))
        face_id = data.pop("face_id")
        image_id = data.pop("image_id")

        flickr_photo_id = id_to_photo.get(image_id)
        if not flickr_photo_id:
            skipped += 1
            continue

        if face_id in existing_ids:
            skipped += 1
            continue

        data["flickr_photo_id"] = flickr_photo_id

        # Denormalize event_name from images
        event_name = id_to_event.get(image_id)
        if event_name:
            data["event_name"] = event_name

        embedding = data.pop("embedding")
        if embedding is not None:
            data["embedding"] = Vector(list(embedding))

        doc_ref = fs_client.collection("face_detections").document(face_id)
        batch.set(doc_ref, data)
        batch_count += 1

        if batch_count >= BATCH_LIMIT:
            batch.commit()
            print(f"  Committed {i + 1}/{len(rows)} face detections")
            time.sleep(SLEEP_SECONDS)
            batch = fs_client.batch()
            batch_count = 0

    if batch_count > 0:
        batch.commit()
    print(f"  Completed: {len(rows) - skipped} face detections uploaded ({skipped} skipped)")


def upload_metadata(
    conn: duckdb.DuckDBPyConnection,
    fs_client: Client,
    dry_run: bool,
) -> None:
    """Upload metadata document with filter options (event names, tag labels)."""
    event_names = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT event_name FROM images ORDER BY event_name"
        ).fetchall()
    ]

    try:
        tag_labels = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT label FROM object_detections ORDER BY label"
            ).fetchall()
        ]
    except duckdb.CatalogException:
        tag_labels = []

    print(f"Metadata: {len(event_names)} event names, {len(tag_labels)} tag labels")

    if dry_run:
        return

    fs_client.collection("metadata").document("filters").set(
        {
            "event_names": event_names,
            "tag_labels": tag_labels,
        }
    )
    print("  Completed: metadata/filters uploaded")


def main() -> None:
    args = get_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: DuckDB file not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"DuckDB: {db_path}")
    print(f"Project: {args.project}")
    if args.dry_run:
        print("Mode: DRY RUN (no data will be uploaded)")
    print()

    conn = duckdb.connect(str(db_path), read_only=True)

    if not args.dry_run:
        fs_client = Client(project=args.project)
    else:
        fs_client = None

    id_to_event = _build_image_event_names(conn)

    id_to_photo = upload_images(conn, fs_client, args.dry_run, force=args.force)
    upload_face_detections(
        conn, fs_client, id_to_photo, id_to_event, args.dry_run, force=args.force
    )
    upload_metadata(conn, fs_client, args.dry_run)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
