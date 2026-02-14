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
        choices=["siglip", "clip"],
        default="siglip",
        help="Model to use: siglip or clip (default: siglip)",
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
        choices=["siglip", "clip"],
        default="siglip",
        help="Model to check status for (default: siglip)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "status":
        _cmd_status(args)
    elif args.command == "generate":
        _cmd_generate(args)


def _resolve_model_config(model_choice: str) -> tuple[str, str]:
    """Return (model_name, db_path) for the chosen model."""
    from pyconjp_image_search.config import (
        CLIP_DB_PATH,
        CLIP_MODEL_NAME,
        DB_PATH,
        SIGLIP_MODEL_NAME,
    )

    if model_choice == "clip":
        return CLIP_MODEL_NAME, str(CLIP_DB_PATH)
    return SIGLIP_MODEL_NAME, str(DB_PATH)


def _cmd_status(args: argparse.Namespace) -> None:
    """Show embedding generation status."""
    from pyconjp_image_search.db import get_connection
    from pyconjp_image_search.embedding.repository import get_embedding_stats

    model_name, db_path = _resolve_model_config(args.model)
    conn = get_connection(db_path)
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

    model_name, db_path = _resolve_model_config(args.model)
    conn = get_connection(db_path)

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
