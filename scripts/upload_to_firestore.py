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
        [--project pyconjp-image-search] [--dry-run] [--force] [--assume-uploaded]

Uploaded document IDs are recorded in the DuckDB table ``firestore_uploads``,
so later runs upload only new documents without reading from Firestore.
"""

import argparse
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import duckdb
from firestore_upload_tracking import ensure_tracking_table, get_uploaded_ids, record_uploaded
from google.cloud.firestore_v1 import Client
from google.cloud.firestore_v1.vector import Vector

SIGLIP_MODEL = "google/siglip2-base-patch16-224"
BATCH_LIMIT = 500  # Firestore batch write limit
SLEEP_SECONDS = 1  # Sleep between batch commits to avoid quota exceeded

IMAGES = "images"
FACE_DETECTIONS = "face_detections"


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
        help="Upload (overwrite) all documents, ignoring upload records",
    )
    parser.add_argument(
        "--assume-uploaded",
        action="store_true",
        help=(
            "When no upload records exist yet, record every document currently in DuckDB"
            " as already uploaded instead of reading existing IDs from Firestore."
            " Use only when Firestore is known to be in sync with DuckDB."
        ),
    )
    parser.add_argument(
        "--update-voronoi",
        action="store_true",
        help="Update only voronoi_partition_ids on existing image documents",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N documents (to resume from where you left off)",
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


def resolve_uploaded_ids(
    conn: duckdb.DuckDBPyConnection,
    fs_client: Client | None,
    collection: str,
    all_doc_ids: list[str],
    *,
    dry_run: bool,
    force: bool,
    assume_uploaded: bool,
) -> set[str]:
    """Return doc IDs to treat as already uploaded (to be skipped).

    Normally this reads only the local DuckDB tracking table. When the table has
    no records for the collection yet, it is bootstrapped once, either from
    Firestore (one read per existing document) or, with --assume-uploaded, by
    marking every current DuckDB document as uploaded (no reads).
    """
    if force:
        print("  Force mode: uploading all documents")
        return set()

    uploaded = get_uploaded_ids(conn, collection)
    if uploaded:
        print(f"  {len(uploaded)} documents recorded as uploaded (no Firestore reads)")
        return uploaded

    if assume_uploaded:
        print(
            f"  No upload records yet: --assume-uploaded marks all {len(all_doc_ids)}"
            " current documents as uploaded"
        )
        if not dry_run:
            record_uploaded(conn, collection, all_doc_ids)
        return set(all_doc_ids)

    if dry_run:
        print(
            "  No upload records yet: the first real run reads all existing IDs from"
            " Firestore once (or use --assume-uploaded)"
        )
        return set()

    assert fs_client is not None
    print("  No upload records yet: reading existing IDs from Firestore once...")
    existing = {d.id for d in fs_client.collection(collection).select([]).stream()}
    record_uploaded(conn, collection, existing)
    print(f"  Recorded {len(existing)} existing documents")
    return existing


def commit_in_batches(
    fs_client: Client,
    conn: duckdb.DuckDBPyConnection,
    collection: str,
    docs: Iterable[tuple[str, dict]],
    total: int,
) -> int:
    """Write docs in batches, recording each committed batch in DuckDB."""
    batch = fs_client.batch()
    batch_ids: list[str] = []
    committed = 0

    def flush() -> None:
        nonlocal batch, batch_ids, committed
        batch.commit()
        record_uploaded(conn, collection, batch_ids)
        committed += len(batch_ids)
        print(f"  Committed {committed}/{total} {collection}")
        batch = fs_client.batch()
        batch_ids = []

    for doc_id, data in docs:
        batch.set(fs_client.collection(collection).document(doc_id), data)
        batch_ids.append(doc_id)
        if len(batch_ids) >= BATCH_LIMIT:
            flush()
            time.sleep(SLEEP_SECONDS)

    if batch_ids:
        flush()
    return committed


def upload_images(
    conn: duckdb.DuckDBPyConnection,
    fs_client: Client | None,
    dry_run: bool,
    *,
    force: bool = False,
    assume_uploaded: bool = False,
) -> dict[int, str]:
    """Upload images + embeddings. Returns mapping of DuckDB image_id -> flickr_photo_id."""
    rows = conn.execute(f"""
        SELECT
            i.id, i.image_url, i.flickr_photo_id, i.album_id, i.album_title,
            i.event_name, i.event_year, i.event_type, i.width, i.height,
            e.embedding, e.voronoi_partition_ids
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
        "voronoi_partition_ids",
    ]

    id_to_photo = {row[0]: row[2] for row in rows}
    print(f"Images: {len(rows)} documents in DuckDB")

    uploaded = resolve_uploaded_ids(
        conn,
        fs_client,
        IMAGES,
        list(id_to_photo.values()),
        dry_run=dry_run,
        force=force,
        assume_uploaded=assume_uploaded,
    )
    targets = [row for row in rows if row[2] not in uploaded]
    print(f"  {len(targets)} documents to upload")

    if dry_run or not targets:
        return id_to_photo
    assert fs_client is not None

    image_tags = _build_image_tags(conn)

    def docs() -> Iterator[tuple[str, dict]]:
        for row in targets:
            data = dict(zip(columns, row))
            image_id = data.pop("id")

            data["embedding"] = Vector(list(data.pop("embedding")))

            partition_ids = data.pop("voronoi_partition_ids")
            if partition_ids is not None:
                data["voronoi_partition_ids"] = list(partition_ids)

            # Denormalize tags from object_detections
            tags = image_tags.get(image_id, [])
            if tags:
                data["tags"] = tags

            yield data["flickr_photo_id"], data

    count = commit_in_batches(fs_client, conn, IMAGES, docs(), len(targets))
    print(f"  Completed: {count} images uploaded ({len(rows) - count} skipped)")
    return id_to_photo


def upload_face_detections(
    conn: duckdb.DuckDBPyConnection,
    fs_client: Client | None,
    id_to_photo: dict[int, str],
    id_to_event: dict[int, str],
    dry_run: bool,
    *,
    force: bool = False,
    assume_uploaded: bool = False,
) -> None:
    """Upload face detections with embeddings."""
    rows = conn.execute("""
        SELECT
            face_id, image_id, model_name,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            det_score, age, gender, embedding,
            person_label, cluster_id, voronoi_partition_ids
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
        "voronoi_partition_ids",
    ]

    # Faces whose image is not uploaded as an image document are never uploaded
    rows = [row for row in rows if row[1] in id_to_photo]
    print(f"Face detections: {len(rows)} documents in DuckDB")

    uploaded = resolve_uploaded_ids(
        conn,
        fs_client,
        FACE_DETECTIONS,
        [row[0] for row in rows],
        dry_run=dry_run,
        force=force,
        assume_uploaded=assume_uploaded,
    )
    targets = [row for row in rows if row[0] not in uploaded]
    print(f"  {len(targets)} documents to upload")

    if dry_run or not targets:
        return
    assert fs_client is not None

    def docs() -> Iterator[tuple[str, dict]]:
        for row in targets:
            data = dict(zip(columns, row))
            face_id = data.pop("face_id")
            image_id = data.pop("image_id")

            data["flickr_photo_id"] = id_to_photo[image_id]

            # Denormalize event_name from images
            event_name = id_to_event.get(image_id)
            if event_name:
                data["event_name"] = event_name

            embedding = data.pop("embedding")
            if embedding is not None:
                data["embedding"] = Vector(list(embedding))

            partition_ids = data.get("voronoi_partition_ids")
            if partition_ids is not None:
                data["voronoi_partition_ids"] = list(partition_ids)

            yield face_id, data

    count = commit_in_batches(fs_client, conn, FACE_DETECTIONS, docs(), len(targets))
    print(f"  Completed: {count} face detections uploaded ({len(rows) - count} skipped)")


def update_image_voronoi(
    conn: duckdb.DuckDBPyConnection,
    fs_client: Client | None,
    dry_run: bool,
    skip: int = 0,
) -> None:
    """Update voronoi_partition_ids on existing image documents."""
    rows = conn.execute(f"""
        SELECT i.flickr_photo_id, e.voronoi_partition_ids
        FROM images i
        JOIN image_embeddings e ON e.image_id = i.id
        WHERE e.model_name = '{SIGLIP_MODEL}'
          AND e.voronoi_partition_ids IS NOT NULL
        ORDER BY i.id
    """).fetchall()

    print(f"Image voronoi updates: {len(rows)} documents")

    if dry_run:
        return
    assert fs_client is not None

    if skip > 0:
        print(f"  Skipping first {skip} documents, resuming from {skip + 1}")
        rows = rows[skip:]

    batch = fs_client.batch()
    batch_count = 0

    for i, (flickr_photo_id, partition_ids) in enumerate(rows):
        doc_ref = fs_client.collection(IMAGES).document(flickr_photo_id)
        batch.update(doc_ref, {"voronoi_partition_ids": list(partition_ids)})
        batch_count += 1

        if batch_count >= BATCH_LIMIT:
            batch.commit()
            print(f"  Updated {skip + i + 1}/{skip + len(rows)} images")
            time.sleep(SLEEP_SECONDS)
            batch = fs_client.batch()
            batch_count = 0

    if batch_count > 0:
        batch.commit()
    print(f"  Completed: {len(rows)} image voronoi_partition_ids updated (skipped {skip})")


def upload_metadata(
    conn: duckdb.DuckDBPyConnection,
    fs_client: Client | None,
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
    assert fs_client is not None

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

    # Writable (except dry-run) so that uploaded IDs can be recorded
    conn = duckdb.connect(str(db_path), read_only=args.dry_run)
    if not args.dry_run:
        ensure_tracking_table(conn)

    fs_client = None if args.dry_run else Client(project=args.project)

    if args.update_voronoi:
        update_image_voronoi(conn, fs_client, args.dry_run, skip=args.skip)
        conn.close()
        print("\nDone!")
        return

    id_to_event = _build_image_event_names(conn)

    id_to_photo = upload_images(
        conn, fs_client, args.dry_run, force=args.force, assume_uploaded=args.assume_uploaded
    )
    upload_face_detections(
        conn,
        fs_client,
        id_to_photo,
        id_to_event,
        args.dry_run,
        force=args.force,
        assume_uploaded=args.assume_uploaded,
    )
    upload_metadata(conn, fs_client, args.dry_run)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
