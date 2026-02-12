"""Image management CLI: download images from Flickr and manage DuckDB metadata."""

import argparse


def main() -> None:
    """CLI entry point for image management."""
    parser = argparse.ArgumentParser(description="PyCon JP image manager")
    subparsers = parser.add_subparsers(dest="command")

    # init-db
    subparsers.add_parser("init-db", help="Initialize the database schema")

    # list-albums
    la_parser = subparsers.add_parser("list-albums", help="List Flickr albums for a user")
    la_parser.add_argument("--user-id", help="Flickr user ID (or set FLICKR_USER_ID in .env)")

    # download-flickr
    dl_parser = subparsers.add_parser(
        "download-flickr", help="Download images from a Flickr album"
    )
    dl_parser.add_argument("--user-id", help="Flickr user ID (or set FLICKR_USER_ID in .env)")
    dl_parser.add_argument("--album-id", required=True, help="Flickr photoset/album ID")
    dl_parser.add_argument("--album-title", help="Album title (auto-detected if omitted)")
    dl_parser.add_argument("--event", required=True, help="Event name")
    dl_parser.add_argument("--year", type=int, required=True, help="Event year")
    dl_parser.add_argument(
        "--event-type", default="conference", help="Event type (default: conference)"
    )
    dl_parser.add_argument(
        "--size",
        default="b",
        help="Photo size: s,q,t,m,z,b,h,k,o (default: b=1024px)",
    )
    dl_parser.add_argument(
        "--dry-run", action="store_true", help="List photos without downloading"
    )

    # list
    list_parser = subparsers.add_parser("list", help="List images in DB")
    list_parser.add_argument("--event", help="Filter by event name")
    list_parser.add_argument("--year", type=int, help="Filter by event year")
    list_parser.add_argument("--album-id", help="Filter by album ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "init-db":
        from pyconjp_image_search.db import get_connection

        conn = get_connection()
        conn.close()
        print("Database initialized successfully.")

    elif args.command == "list-albums":
        _cmd_list_albums(args)

    elif args.command == "download-flickr":
        _cmd_download_flickr(args)

    elif args.command == "list":
        from pyconjp_image_search.db import get_connection
        from pyconjp_image_search.manager.repository import list_images

        conn = get_connection()
        images = list_images(
            conn,
            event_name=args.event,
            event_year=args.year,
            album_id=args.album_id,
        )
        conn.close()
        for img in images:
            path_info = img.relative_path or img.image_url
            print(f"[{img.event_name} {img.event_year}] {path_info}")


def _resolve_user_id(args: argparse.Namespace) -> str | None:
    """Resolve Flickr user ID from CLI args or .env."""
    from pyconjp_image_search.config import FLICKR_USER_ID

    user_id = getattr(args, "user_id", None) or FLICKR_USER_ID
    if not user_id:
        print("Error: --user-id required (or set FLICKR_USER_ID in .env)")
        return None
    return user_id


def _cmd_list_albums(args: argparse.Namespace) -> None:
    """List Flickr albums for a user."""
    from pyconjp_image_search.manager.flickr_client import FlickrClient

    user_id = _resolve_user_id(args)
    if not user_id:
        return

    client = FlickrClient()
    albums = client.list_albums(user_id)
    for album in albums:
        print(f"  {album.id}  {album.count_photos:>5} photos  {album.title}")


def _cmd_download_flickr(args: argparse.Namespace) -> None:
    """Download images from a Flickr album."""
    from pyconjp_image_search.db import get_connection
    from pyconjp_image_search.manager.downloader import download_album
    from pyconjp_image_search.manager.flickr_client import FlickrClient
    from pyconjp_image_search.manager.repository import (
        get_existing_photo_ids,
        insert_images,
    )

    user_id = _resolve_user_id(args)
    if not user_id:
        return

    client = FlickrClient()

    # Auto-detect album title if not provided
    album_title = args.album_title
    if not album_title:
        albums = client.list_albums(user_id)
        for a in albums:
            if a.id == args.album_id:
                album_title = a.title
                break
        if not album_title:
            album_title = args.album_id

    if args.dry_run:
        photos = client.get_all_photos_in_album(args.album_id, user_id)
        print(f"Album '{album_title}' has {len(photos)} photos (dry run, nothing downloaded)")
        return

    conn = get_connection()
    existing = get_existing_photo_ids(conn, album_id=args.album_id)

    metadata_list = download_album(
        client=client,
        user_id=user_id,
        album_id=args.album_id,
        album_title=album_title,
        event_name=args.event,
        event_year=args.year,
        event_type=args.event_type,
        size=args.size,
        existing_photo_ids=existing,
    )

    insert_images(conn, metadata_list)
    conn.close()
    print(f"Downloaded and registered {len(metadata_list)} new images from '{album_title}'.")
