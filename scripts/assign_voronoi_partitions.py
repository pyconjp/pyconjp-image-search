# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb>=1.0",
#     "numpy",
# ]
# ///
"""Assign Voronoi partition IDs to face detections in DuckDB.

Reads pre-computed centroids from data/voronoi_pivots.json, computes cosine
similarity between each face embedding and the 256 centroids, and assigns
the top-N nearest partition IDs to each face.

Usage:
    cd /path/to/pyconjp-image-search
    uv run scripts/assign_voronoi_partitions.py \
        [--db pyconjp_image_search.duckdb] \
        [--centroids data/voronoi_pivots.json] \
        [--n-assign 2] [--force]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

# Add project root to path for schema import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pyconjp_image_search.manager.schema import ensure_schema

FACE_MODEL_NAME = "insightface/buffalo_l"


def load_centroids(centroids_path: Path) -> np.ndarray:
    """Load centroids from voronoi_pivots.json."""
    with open(centroids_path) as f:
        data = json.load(f)
    centroids = np.array(data["centroids"], dtype=np.float32)
    n_pivots = data["n_pivots"]
    dim = data["dim"]
    assert centroids.shape == (n_pivots, dim), (
        f"Expected ({n_pivots}, {dim}), got {centroids.shape}"
    )
    # L2 normalize
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-8)
    print(f"Loaded {n_pivots} centroids (dim={dim}) from {centroids_path}")
    return centroids


def fetch_face_embeddings(conn: duckdb.DuckDBPyConnection) -> tuple[list[str], np.ndarray]:
    """Fetch all face embeddings from DuckDB."""
    result = conn.execute(
        f"SELECT face_id, embedding FROM face_detections WHERE model_name = '{FACE_MODEL_NAME}'"
    ).fetchall()
    face_ids = [row[0] for row in result]
    embeddings = np.array([list(row[1]) for row in result], dtype=np.float32)
    print(f"Fetched {len(face_ids)} face embeddings ({embeddings.shape})")
    return face_ids, embeddings


def compute_assignments(
    embeddings: np.ndarray, centroids: np.ndarray, n_assign: int
) -> np.ndarray:
    """Compute top-N nearest centroid IDs for each embedding.

    Returns array of shape (n_faces, n_assign) with partition IDs.
    """
    # L2 normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / np.maximum(norms, 1e-8)

    # Cosine similarity = dot product of normalized vectors
    # Shape: (n_faces, n_centroids)
    similarities = embeddings_normed @ centroids.T

    # Top-N indices per face
    # Use argpartition for efficiency, then sort the top-N
    top_indices = np.argpartition(-similarities, n_assign, axis=1)[:, :n_assign]
    # Sort within top-N by similarity (descending)
    for i in range(len(top_indices)):
        row_sims = similarities[i, top_indices[i]]
        sorted_order = np.argsort(-row_sims)
        top_indices[i] = top_indices[i, sorted_order]

    return top_indices


def update_database(
    conn: duckdb.DuckDBPyConnection,
    face_ids: list[str],
    assignments: np.ndarray,
    force: bool,
) -> None:
    """Update voronoi_partition_ids in face_detections table."""
    # Check if already assigned
    if not force:
        count = conn.execute(
            "SELECT COUNT(*) FROM face_detections WHERE voronoi_partition_ids IS NOT NULL"
        ).fetchone()[0]
        if count > 0:
            print(f"Already {count} faces with partition IDs. Use --force to reassign.")
            return

    batch_size = 1000
    total = len(face_ids)
    start = time.time()

    for i in range(0, total, batch_size):
        batch_ids = face_ids[i : i + batch_size]
        batch_assignments = assignments[i : i + batch_size]

        for face_id, partition_ids in zip(batch_ids, batch_assignments):
            ids_list = partition_ids.tolist()
            conn.execute(
                "UPDATE face_detections SET voronoi_partition_ids = ? WHERE face_id = ?",
                [ids_list, face_id],
            )

        elapsed = time.time() - start
        done = min(i + batch_size, total)
        print(f"  Updated {done}/{total} faces ({elapsed:.1f}s)")

    # Verify
    count = conn.execute(
        "SELECT COUNT(*) FROM face_detections WHERE voronoi_partition_ids IS NOT NULL"
    ).fetchone()[0]
    print(f"Verified: {count} faces now have voronoi_partition_ids")


def print_partition_stats(conn: duckdb.DuckDBPyConnection) -> None:
    """Print partition distribution statistics."""
    result = conn.execute("""
        WITH expanded AS (
            SELECT unnest(voronoi_partition_ids) AS pid
            FROM face_detections
            WHERE voronoi_partition_ids IS NOT NULL
        )
        SELECT pid, COUNT(*) AS cnt
        FROM expanded
        GROUP BY pid
        ORDER BY cnt DESC
    """).fetchall()

    if not result:
        print("No partition data found.")
        return

    counts = [row[1] for row in result]
    print(f"\nPartition stats ({len(result)} partitions used):")
    print(f"  Total assignments: {sum(counts)}")
    print(f"  Min faces/partition: {min(counts)}")
    print(f"  Max faces/partition: {max(counts)}")
    print(f"  Mean faces/partition: {sum(counts) / len(counts):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign Voronoi partitions to face detections")
    parser.add_argument("--db", default="pyconjp_image_search.duckdb", help="DuckDB file path")
    parser.add_argument(
        "--centroids", default="data/voronoi_pivots.json", help="Centroids JSON file"
    )
    parser.add_argument("--n-assign", type=int, default=2, help="Number of partitions per face")
    parser.add_argument("--force", action="store_true", help="Force reassignment")
    args = parser.parse_args()

    db_path = Path(args.db)
    centroids_path = Path(args.centroids)

    if not db_path.exists():
        print(f"Error: DB file not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    if not centroids_path.exists():
        print(f"Error: Centroids file not found: {centroids_path}", file=sys.stderr)
        sys.exit(1)

    # Load centroids
    centroids = load_centroids(centroids_path)

    # Connect to DuckDB
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)

    # Fetch face embeddings
    face_ids, embeddings = fetch_face_embeddings(conn)
    if len(face_ids) == 0:
        print("No face embeddings found.")
        conn.close()
        return

    # Compute assignments
    print(f"Computing top-{args.n_assign} partition assignments...")
    t0 = time.time()
    assignments = compute_assignments(embeddings, centroids, args.n_assign)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Update database
    print("Updating database...")
    update_database(conn, face_ids, assignments, args.force)

    # Print stats
    print_partition_stats(conn)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
