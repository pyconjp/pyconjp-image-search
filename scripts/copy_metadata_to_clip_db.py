"""Copy image metadata from the SigLIP DB to the CLIP DB (no embeddings)."""

import duckdb

from pyconjp_image_search.config import CLIP_DB_PATH, DB_PATH
from pyconjp_image_search.manager.schema import ensure_schema


def main() -> None:
    """Copy images table from existing DB to a new CLIP DB."""
    print(f"Source DB: {DB_PATH}")
    print(f"Target DB: {CLIP_DB_PATH}")

    # Create target DB with schema
    target = duckdb.connect(str(CLIP_DB_PATH))
    ensure_schema(target)

    # Attach source DB as read-only
    target.execute(f"ATTACH '{DB_PATH}' AS source (READ_ONLY)")

    # Copy images (skip duplicates for idempotency)
    target.execute("""
        INSERT INTO images
        SELECT * FROM source.images
        ON CONFLICT (image_url) DO NOTHING
    """)

    row = target.execute("SELECT COUNT(*) FROM images").fetchone()
    count = row[0] if row else 0
    print(f"CLIP DB now has {count} images at {CLIP_DB_PATH}")

    target.execute("DETACH source")
    target.close()
    print("Done.")


if __name__ == "__main__":
    main()
