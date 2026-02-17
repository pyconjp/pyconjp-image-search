"""Embedding CLI: generate and manage embeddings."""

import argparse


def main() -> None:
    """CLI entry point for embedding operations."""
    parser = argparse.ArgumentParser(description="PyCon JP image embedding")
    subparsers = parser.add_subparsers(dest="command")

    # generate
    gen_parser = subparsers.add_parser(
        "generate", help="Generate embeddings for unprocessed images"
    )
    gen_parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    gen_parser.add_argument("--device", default="cuda", help="Device: cuda or cpu (default: cuda)")
    gen_parser.add_argument(
        "--model",
        choices=["siglip", "siglip-large", "clip"],
        default="siglip",
        help="Model to use: siglip, siglip-large, or clip (default: siglip)",
    )
    gen_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to embed (default: all)",
    )
    gen_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate all embeddings (overwrite existing)",
    )

    # status
    status_parser = subparsers.add_parser("status", help="Show embedding generation status")
    status_parser.add_argument(
        "--model",
        choices=["siglip", "siglip-large", "clip"],
        default="siglip",
        help="Model to check status for (default: siglip)",
    )

    # face-generate
    face_gen_parser = subparsers.add_parser(
        "face-generate", help="Detect faces and generate face embeddings"
    )
    face_gen_parser.add_argument(
        "--device", default="cuda", help="Device: cuda or cpu (default: cuda)"
    )
    face_gen_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to process (default: all)",
    )
    face_gen_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process all images (delete existing face detections)",
    )
    face_gen_parser.add_argument(
        "--commit-interval",
        type=int,
        default=100,
        help="Commit to DB every N images (default: 100)",
    )

    # face-status
    subparsers.add_parser("face-status", help="Show face detection status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "status":
        _cmd_status(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "face-generate":
        _cmd_face_generate(args)
    elif args.command == "face-status":
        _cmd_face_status()


def _resolve_model_config(model_choice: str) -> tuple[str, str, int]:
    """Return (model_name, db_path, embedding_dim) for the chosen model."""
    from pyconjp_image_search.config import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_choice]
    return cfg["model_name"], str(cfg["db_path"]), cfg["embedding_dim"]


def _cmd_status(args: argparse.Namespace) -> None:
    """Show embedding generation status."""
    from pyconjp_image_search.db import get_connection
    from pyconjp_image_search.embedding.repository import get_embedding_stats

    model_name, db_path, embedding_dim = _resolve_model_config(args.model)
    conn = get_connection(db_path, embedding_dim=embedding_dim)
    total, embedded = get_embedding_stats(conn, model_name)
    conn.close()
    print(f"Model: {model_name}")
    print(f"DB: {db_path}")
    print(f"Embedded: {embedded}/{total} images")
    if total > 0:
        print(f"Progress: {embedded / total * 100:.1f}%")


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate embeddings for unprocessed images."""
    from pathlib import Path

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    from pyconjp_image_search.config import DATA_DIR
    from pyconjp_image_search.db import get_connection
    from pyconjp_image_search.embedding.repository import (
        get_all_image_ids,
        get_unembedded_image_ids,
        insert_embeddings,
    )

    model_name, db_path, embedding_dim = _resolve_model_config(args.model)
    conn = get_connection(db_path, embedding_dim=embedding_dim)

    if args.force:
        unembedded = get_all_image_ids(conn)
        print(f"Force mode: re-generating all {len(unembedded)} embeddings.")
    else:
        unembedded = get_unembedded_image_ids(conn, model_name)
        if not unembedded:
            print("All images already have embeddings.")
            conn.close()
            return

    if args.limit is not None:
        unembedded = unembedded[: args.limit]

    print(f"Found {len(unembedded)} images to embed.")
    print(f"Loading model: {model_name} on {args.device}...")

    if args.model == "clip":
        from pyconjp_image_search.embedding.clip import CLIPEmbedder

        embedder = CLIPEmbedder(model_name=model_name, device=args.device)
    else:
        # Both siglip and siglip-large use the same SigLIPEmbedder class
        from pyconjp_image_search.embedding.siglip import SigLIPEmbedder

        embedder = SigLIPEmbedder(model_name=model_name, device=args.device)

    batch_size = args.batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Generating embeddings", total=len(unembedded))

        for start in range(0, len(unembedded), batch_size):
            batch = unembedded[start : start + batch_size]
            image_ids = [item[0] for item in batch]
            image_paths = [DATA_DIR / item[1] for item in batch]

            # Skip missing files
            valid_ids = []
            valid_paths: list[Path] = []
            for img_id, img_path in zip(image_ids, image_paths):
                if img_path.exists():
                    valid_ids.append(img_id)
                    valid_paths.append(img_path)

            if valid_paths:
                embeddings = embedder.embed_images(valid_paths)
                insert_embeddings(conn, valid_ids, embeddings, model_name)

            progress.advance(task, advance=len(batch))

    conn.close()
    print(f"Done. Embedded {len(unembedded)} images.")


def _cmd_face_generate(args: argparse.Namespace) -> None:
    """Detect faces and generate face embeddings for all images."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    from pyconjp_image_search.config import CLIP_DB_PATH, DATA_DIR, INSIGHTFACE_MODEL_NAME
    from pyconjp_image_search.db import get_connection
    from pyconjp_image_search.embedding.face_repository import (
        get_face_processed_image_ids,
        insert_face_detections,
        mark_image_processed,
    )
    from pyconjp_image_search.embedding.insightface_embedder import InsightFaceEmbedder
    from pyconjp_image_search.embedding.repository import get_all_image_ids

    conn = get_connection(str(CLIP_DB_PATH))

    all_images = get_all_image_ids(conn)

    if args.force:
        conn.execute("DELETE FROM face_detections WHERE model_name = ?", [INSIGHTFACE_MODEL_NAME])
        conn.execute(
            "DELETE FROM face_processed_images WHERE model_name = ?", [INSIGHTFACE_MODEL_NAME]
        )
        conn.commit()
        pending = all_images
        print(f"Force mode: re-processing all {len(pending)} images.")
    else:
        processed = get_face_processed_image_ids(conn, INSIGHTFACE_MODEL_NAME)
        pending = [(img_id, path) for img_id, path in all_images if img_id not in processed]
        if not pending:
            print("All images already have face detections.")
            conn.close()
            return

    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Found {len(pending)} images to process.")
    print(f"Loading InsightFace model on {args.device}...")

    embedder = InsightFaceEmbedder(device=args.device)
    commit_interval = args.commit_interval
    total_faces = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Detecting faces", total=len(pending))

        for idx, (image_id, relative_path) in enumerate(pending):
            image_path = DATA_DIR / relative_path
            if not image_path.exists():
                progress.advance(task)
                continue

            try:
                detections = embedder.detect_faces(image_path, image_id)
                if detections:
                    insert_face_detections(conn, detections)
                    total_faces += len(detections)
                mark_image_processed(conn, image_id, INSIGHTFACE_MODEL_NAME, len(detections))
            except Exception:
                errors += 1

            if (idx + 1) % commit_interval == 0:
                conn.commit()

            progress.advance(task)

    conn.commit()
    conn.close()
    print("\nDone.")
    print(f"  Images processed: {len(pending)}")
    print(f"  Faces detected: {total_faces}")
    if errors > 0:
        print(f"  Errors: {errors}")


def _cmd_face_status() -> None:
    """Show face detection status."""
    from pyconjp_image_search.config import CLIP_DB_PATH, INSIGHTFACE_MODEL_NAME
    from pyconjp_image_search.db import get_connection
    from pyconjp_image_search.embedding.face_repository import get_face_stats

    conn = get_connection(str(CLIP_DB_PATH))
    total, processed, faces = get_face_stats(conn, INSIGHTFACE_MODEL_NAME)
    conn.close()
    print(f"Model: {INSIGHTFACE_MODEL_NAME}")
    print(f"DB: {CLIP_DB_PATH}")
    print(f"Processed: {processed}/{total} images")
    print(f"Faces detected: {faces}")
    if processed > 0:
        print(f"Average faces per image: {faces / processed:.1f}")
    if total > 0:
        print(f"Progress: {processed / total * 100:.1f}%")
