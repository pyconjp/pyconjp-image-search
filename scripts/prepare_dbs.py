"""Prepare DuckDB files for SigLIP 2 base and Large models.

1. SigLIP 2 base DB: delete old v1 embeddings, copy face data from CLIP DB
2. SigLIP 2 Large DB: create new DB, copy metadata + face data
3. Create symlinks in web/public/
"""

import os

import duckdb

from pyconjp_image_search.config import (
    CLIP_DB_PATH,
    DB_PATH,
    SIGLIP_LARGE_DB_PATH,
    SIGLIP_LARGE_EMBEDDING_DIM,
)
from pyconjp_image_search.manager.schema import ensure_schema

PROJECT_ROOT = DB_PATH.parent


def prepare_siglip_base_db() -> None:
    """Prepare the SigLIP 2 base DB: clear old embeddings, copy face data."""
    print(f"\n=== Preparing SigLIP 2 base DB: {DB_PATH} ===")

    conn = duckdb.connect(str(DB_PATH))
    ensure_schema(conn, embedding_dim=768)

    # Delete old SigLIP v1 embeddings (model_name != siglip2)
    deleted = conn.execute(
        "DELETE FROM image_embeddings WHERE model_name NOT LIKE '%siglip2%'"
    ).fetchone()
    print(f"Deleted old embeddings: {deleted}")

    # Check if face data already exists
    face_count_row = conn.execute("SELECT COUNT(*) FROM face_detections").fetchone()
    face_count = face_count_row[0] if face_count_row else 0

    if face_count > 0:
        print(f"Face data already exists ({face_count} faces). Skipping copy.")
    else:
        # Copy face data from CLIP DB
        if not CLIP_DB_PATH.exists():
            print(f"CLIP DB not found at {CLIP_DB_PATH}. Skipping face data copy.")
        else:
            print(f"Copying face data from {CLIP_DB_PATH}...")
            conn.execute(f"ATTACH '{CLIP_DB_PATH}' AS clip_db (READ_ONLY)")
            conn.execute("""
                INSERT INTO face_detections
                SELECT * FROM clip_db.face_detections
                ON CONFLICT (face_id) DO NOTHING
            """)
            conn.execute("""
                INSERT INTO face_processed_images
                SELECT * FROM clip_db.face_processed_images
                ON CONFLICT (image_id, model_name) DO NOTHING
            """)
            conn.execute("DETACH clip_db")

            face_count_row = conn.execute("SELECT COUNT(*) FROM face_detections").fetchone()
            print(f"Face detections copied: {face_count_row[0] if face_count_row else 0}")

    conn.close()
    print("SigLIP 2 base DB ready.")


def prepare_siglip_large_db() -> None:
    """Create the SigLIP 2 Large DB with metadata and face data."""
    print(f"\n=== Preparing SigLIP 2 Large DB: {SIGLIP_LARGE_DB_PATH} ===")

    conn = duckdb.connect(str(SIGLIP_LARGE_DB_PATH))
    ensure_schema(conn, embedding_dim=SIGLIP_LARGE_EMBEDDING_DIM)

    # Check if images already exist
    img_count_row = conn.execute("SELECT COUNT(*) FROM images").fetchone()
    img_count = img_count_row[0] if img_count_row else 0

    if img_count > 0:
        print(f"Images already exist ({img_count}). Skipping metadata copy.")
    else:
        # Copy images from base DB
        if not DB_PATH.exists():
            print(f"Base DB not found at {DB_PATH}. Cannot copy metadata.")
            conn.close()
            return

        print(f"Copying image metadata from {DB_PATH}...")
        conn.execute(f"ATTACH '{DB_PATH}' AS source (READ_ONLY)")
        conn.execute("""
            INSERT INTO images
            SELECT * FROM source.images
            ON CONFLICT (image_url) DO NOTHING
        """)
        conn.execute("DETACH source")

        # Update sequence to avoid ID conflicts
        max_id_row = conn.execute("SELECT MAX(id) FROM images").fetchone()
        max_id = max_id_row[0] if max_id_row and max_id_row[0] else 0
        if max_id > 0:
            conn.execute("DROP SEQUENCE IF EXISTS images_id_seq CASCADE")
            conn.execute(f"CREATE SEQUENCE images_id_seq START {max_id + 1}")

        img_count_row = conn.execute("SELECT COUNT(*) FROM images").fetchone()
        print(f"Images copied: {img_count_row[0] if img_count_row else 0}")

    # Copy face data from CLIP DB
    face_count_row = conn.execute("SELECT COUNT(*) FROM face_detections").fetchone()
    face_count = face_count_row[0] if face_count_row else 0

    if face_count > 0:
        print(f"Face data already exists ({face_count} faces). Skipping copy.")
    else:
        if not CLIP_DB_PATH.exists():
            print(f"CLIP DB not found at {CLIP_DB_PATH}. Skipping face data copy.")
        else:
            print(f"Copying face data from {CLIP_DB_PATH}...")
            conn.execute(f"ATTACH '{CLIP_DB_PATH}' AS clip_db (READ_ONLY)")
            conn.execute("""
                INSERT INTO face_detections
                SELECT * FROM clip_db.face_detections
                ON CONFLICT (face_id) DO NOTHING
            """)
            conn.execute("""
                INSERT INTO face_processed_images
                SELECT * FROM clip_db.face_processed_images
                ON CONFLICT (image_id, model_name) DO NOTHING
            """)
            conn.execute("DETACH clip_db")

            face_count_row = conn.execute("SELECT COUNT(*) FROM face_detections").fetchone()
            print(f"Face detections copied: {face_count_row[0] if face_count_row else 0}")

    conn.close()
    print("SigLIP 2 Large DB ready.")


def create_web_symlinks() -> None:
    """Create symlinks in web/public/ for all DuckDB files."""
    print("\n=== Creating web/public/ symlinks ===")
    public_dir = PROJECT_ROOT / "web" / "public"
    public_dir.mkdir(parents=True, exist_ok=True)

    symlinks = {
        "pyconjp_image_search.duckdb": DB_PATH,
        "pyconjp_image_search_siglip2_large.duckdb": SIGLIP_LARGE_DB_PATH,
        "pyconjp_image_search_clip.duckdb": CLIP_DB_PATH,
    }

    for name, target in symlinks.items():
        link_path = public_dir / name
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_symlink() and os.readlink(str(link_path)) == str(target):
                print(f"  {name} -> already correct")
                continue
            link_path.unlink()
        os.symlink(str(target), str(link_path))
        print(f"  {name} -> {target}")

    print("Symlinks ready.")


def main() -> None:
    """Run all DB preparation steps."""
    print("PyCon JP Image Search - DB Preparation")
    print("=" * 50)

    prepare_siglip_base_db()
    prepare_siglip_large_db()
    create_web_symlinks()

    print("\n" + "=" * 50)
    print("All databases prepared successfully.")
    print("\nNext steps:")
    print("  1. uv run pyconjp-embed generate --model siglip --force")
    print("  2. uv run pyconjp-embed generate --model siglip-large")


if __name__ == "__main__":
    main()
