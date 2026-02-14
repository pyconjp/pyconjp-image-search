"""Fix event_name inconsistencies in both DuckDB databases.

Updates event names and years to correct naming variations
originating from Flickr album titles.

DuckDB does not allow UPDATE on a table referenced by a foreign key,
so we temporarily drop and recreate the image_embeddings table.
"""

import argparse

import duckdb

from pyconjp_image_search.config import CLIP_DB_PATH, DB_PATH

# Mapping: (old_event_name) -> (new_event_name, new_event_year)
# new_event_year=None means keep existing year unchanged.
FIXES: list[tuple[str, str, int | None]] = [
    ("Pycon JP 2019", "PyCon JP 2019", None),
    ("PyCon APAC Tutorial, Prepare, WelcomeParty", "PyCon APAC 2013", 2013),
    ("PyCon mini JP", "PyCon mini JP 2011", 2011),
    ("PyCon US 2025", "PyCon US 2025 report", None),
    ("Guido meetup", "Guido meetup 2023", 2023),
]


def show_events(conn: duckdb.DuckDBPyConnection) -> None:
    rows = conn.execute(
        "SELECT event_name, event_year, COUNT(*) AS cnt "
        "FROM images GROUP BY event_name, event_year "
        "ORDER BY event_name, event_year"
    ).fetchall()
    for name, year, cnt in rows:
        print(f"  {name:50s} year={year}  count={cnt}")


def _fetchone_scalar(conn: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row else 0


def apply_fixes(db_path: str, *, dry_run: bool) -> None:
    label = "DRY RUN" if dry_run else "APPLY"
    print(f"\n{'='*60}")
    print(f"[{label}] Database: {db_path}")
    print(f"{'='*60}")

    conn = duckdb.connect(str(db_path), read_only=dry_run)

    print("\n--- Before ---")
    show_events(conn)

    if dry_run:
        for old_name, new_name, new_year in FIXES:
            if new_year is not None:
                sql = ("UPDATE images SET event_name = ?, "
                       "event_year = ? WHERE event_name = ?")
                params = [new_name, new_year, old_name]
            else:
                sql = ("UPDATE images SET event_name = ? "
                       "WHERE event_name = ?")
                params = [new_name, old_name]
            print(f"  [DRY RUN] {sql}  params={params}")
    else:
        # Temporarily drop FK constraint by backing up
        # and recreating image_embeddings
        has_embeddings = _fetchone_scalar(
            conn, "SELECT COUNT(*) FROM image_embeddings"
        ) > 0

        print("  Backing up image_embeddings...")
        conn.execute(
            "CREATE TEMPORARY TABLE _emb_backup "
            "AS SELECT * FROM image_embeddings"
        )
        conn.execute("DROP TABLE image_embeddings")

        # Now we can UPDATE images freely
        for old_name, new_name, new_year in FIXES:
            if new_year is not None:
                row = conn.execute(
                    "UPDATE images SET event_name = ?, "
                    "event_year = ? WHERE event_name = ?",
                    [new_name, new_year, old_name],
                ).fetchone()
            else:
                row = conn.execute(
                    "UPDATE images SET event_name = ? "
                    "WHERE event_name = ?",
                    [new_name, old_name],
                ).fetchone()
            cnt = int(row[0]) if row else 0
            suffix = f" (year={new_year})" if new_year is not None else ""
            print(f"  Updated {cnt} rows: '{old_name}' -> '{new_name}'"
                  + suffix)

        # Restore image_embeddings with FK
        print("  Restoring image_embeddings...")
        conn.execute("""
            CREATE TABLE image_embeddings (
                image_id    INTEGER NOT NULL,
                model_name  VARCHAR NOT NULL,
                embedding   FLOAT[768],
                created_at  TIMESTAMP DEFAULT current_timestamp,
                PRIMARY KEY (image_id, model_name),
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        conn.execute(
            "INSERT INTO image_embeddings "
            "SELECT * FROM _emb_backup"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_model "
            "ON image_embeddings(model_name)"
        )
        conn.execute("DROP TABLE _emb_backup")

        restored = _fetchone_scalar(
            conn, "SELECT COUNT(*) FROM image_embeddings"
        )
        print(f"  Restored {restored} embeddings"
              f" (had_data={has_embeddings})")

        print("\n--- After ---")
        show_events(conn)

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix event name inconsistencies in DuckDB")
    parser.add_argument("--dry-run", action="store_true", help="Show SQL without executing")
    args = parser.parse_args()

    for db_path in [DB_PATH, CLIP_DB_PATH]:
        apply_fixes(str(db_path), dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
