# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb>=1.0",
#     "numpy",
# ]
# ///
"""Assign Voronoi partition IDs to image embeddings in DuckDB.

Reads pre-computed centroids from data/image_voronoi_pivots.json, computes
cosine similarity between each image embedding and the centroids, and assigns
the top-N nearest partition IDs to each image.

Usage:
    cd /path/to/pyconjp-image-search
    uv run scripts/assign_image_voronoi_partitions.py \
        [--db pyconjp_image_search.duckdb] \
        [--centroids data/image_voronoi_pivots.json] \
        [--model-name google/siglip2-base-patch16-224] \
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


def load_centroids(centroids_path: Path) -> np.ndarray:
    """Load centroids from image_voronoi_pivots.json."""
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


def fetch_image_embeddings(
    conn: duckdb.DuckDBPyConnection, model_name: str
) -> tuple[list[int], np.ndarray]:
    """Fetch all image embeddings for a given model from DuckDB."""
    result = conn.execute(
        "SELECT image_id, embedding FROM image_embeddings WHERE model_name = ?",
        [model_name],
    ).fetchall()
    image_ids = [row[0] for row in result]
    embeddings = np.array([list(row[1]) for row in result], dtype=np.float32)
    print(
        f"Fetched {len(image_ids)} image embeddings ({embeddings.shape}) for model '{model_name}'"
    )
    return image_ids, embeddings


def compute_assignments(
    embeddings: np.ndarray, centroids: np.ndarray, n_assign: int
) -> np.ndarray:
    """Compute top-N nearest centroid IDs for each embedding.

    Returns array of shape (n_images, n_assign) with partition IDs.
    """
    # L2 normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / np.maximum(norms, 1e-8)

    # Cosine similarity = dot product of normalized vectors
    # Shape: (n_images, n_centroids)
    similarities = embeddings_normed @ centroids.T

    # Top-N indices per image
    top_indices = np.argpartition(-similarities, n_assign, axis=1)[:, :n_assign]
    # Sort within top-N by similarity (descending)
    for i in range(len(top_indices)):
        row_sims = similarities[i, top_indices[i]]
        sorted_order = np.argsort(-row_sims)
        top_indices[i] = top_indices[i, sorted_order]

    return top_indices


def update_database(
    conn: duckdb.DuckDBPyConnection,
    image_ids: list[int],
    model_name: str,
    assignments: np.ndarray,
    force: bool,
) -> None:
    """Update voronoi_partition_ids in image_embeddings table."""
    if not force:
        row = conn.execute(
            "SELECT COUNT(*) FROM image_embeddings"
            " WHERE model_name = ? AND voronoi_partition_ids IS NOT NULL",
            [model_name],
        ).fetchone()
        count = row[0] if row else 0
        if count > 0:
            print(f"Already {count} embeddings with partition IDs. Use --force to reassign.")
            return

    batch_size = 1000
    total = len(image_ids)
    start = time.time()

    for i in range(0, total, batch_size):
        batch_ids = image_ids[i : i + batch_size]
        batch_assignments = assignments[i : i + batch_size]

        for image_id, partition_ids in zip(batch_ids, batch_assignments):
            ids_list = partition_ids.tolist()
            conn.execute(
                "UPDATE image_embeddings SET voronoi_partition_ids = ?"
                " WHERE image_id = ? AND model_name = ?",
                [ids_list, image_id, model_name],
            )

        elapsed = time.time() - start
        done = min(i + batch_size, total)
        print(f"  Updated {done}/{total} embeddings ({elapsed:.1f}s)")

    # Verify
    row = conn.execute(
        "SELECT COUNT(*) FROM image_embeddings"
        " WHERE model_name = ? AND voronoi_partition_ids IS NOT NULL",
        [model_name],
    ).fetchone()
    count = row[0] if row else 0
    print(f"Verified: {count} embeddings now have voronoi_partition_ids")


def print_partition_stats(conn: duckdb.DuckDBPyConnection, model_name: str) -> None:
    """Print partition distribution statistics."""
    result = conn.execute(
        """
        WITH expanded AS (
            SELECT unnest(voronoi_partition_ids) AS pid
            FROM image_embeddings
            WHERE model_name = ? AND voronoi_partition_ids IS NOT NULL
        )
        SELECT pid, COUNT(*) AS cnt
        FROM expanded
        GROUP BY pid
        ORDER BY cnt DESC
    """,
        [model_name],
    ).fetchall()

    if not result:
        print("No partition data found.")
        return

    counts = [row[1] for row in result]
    print(f"\nPartition stats ({len(result)} partitions used):")
    print(f"  Total assignments: {sum(counts)}")
    print(f"  Min images/partition: {min(counts)}")
    print(f"  Max images/partition: {max(counts)}")
    print(f"  Mean images/partition: {sum(counts) / len(counts):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign Voronoi partitions to image embeddings")
    parser.add_argument("--db", default="pyconjp_image_search.duckdb", help="DuckDB file path")
    parser.add_argument(
        "--centroids", default="data/image_voronoi_pivots.json", help="Centroids JSON file"
    )
    parser.add_argument(
        "--model-name",
        default="google/siglip2-base-patch16-224",
        help="Model name in image_embeddings table",
    )
    parser.add_argument("--n-assign", type=int, default=2, help="Number of partitions per image")
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

    # Fetch image embeddings
    image_ids, embeddings = fetch_image_embeddings(conn, args.model_name)
    if len(image_ids) == 0:
        print("No image embeddings found.")
        conn.close()
        return

    # Compute assignments
    print(f"Computing top-{args.n_assign} partition assignments...")
    t0 = time.time()
    assignments = compute_assignments(embeddings, centroids, args.n_assign)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Update database
    print("Updating database...")
    update_database(conn, image_ids, args.model_name, assignments, args.force)

    # Print stats
    print_partition_stats(conn, args.model_name)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
