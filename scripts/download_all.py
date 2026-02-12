"""Download all Flickr albums for PyCon JP."""

import re
import time

from pyconjp_image_search.config import FLICKR_USER_ID
from pyconjp_image_search.db import get_connection
from pyconjp_image_search.manager.downloader import download_album
from pyconjp_image_search.manager.flickr_client import FlickrClient
from pyconjp_image_search.manager.repository import (
    get_existing_photo_ids,
    insert_images,
)


def parse_event_info(title: str) -> tuple[str, int]:
    """Parse event name and year from album title.

    Examples:
        'PyCon JP 2025 Conference Day1' -> ('PyCon JP 2025', 2025)
        'PyCon APAC 2023 Day 2' -> ('PyCon APAC 2023', 2023)
        'PyCon mini 東海 2025' -> ('PyCon mini 東海 2025', 2025)
        'PyCon JP 2012 Day1' -> ('PyCon JP 2012', 2012)
        'Guido meetup' -> ('Guido meetup', 0)
    """
    match = re.search(r"(20\d{2})", title)
    if match:
        year = int(match.group(1))
        # Event name = everything up to and including the year
        idx = match.end()
        event_name = title[:idx].strip()
        return event_name, year
    return title.strip(), 0


def main() -> None:
    client = FlickrClient()
    user_id = FLICKR_USER_ID
    if not user_id:
        print("Error: FLICKR_USER_ID not set in .env")
        return

    print("Fetching album list...")
    albums = client.list_albums(user_id)
    print(f"Found {len(albums)} albums\n")

    conn = get_connection()
    total_downloaded = 0

    for i, album in enumerate(albums, 1):
        event_name, year = parse_event_info(album.title)
        print(f"[{i}/{len(albums)}] {album.title} ({album.count_photos} photos)")
        print(f"  Event: {event_name}, Year: {year}")

        existing = get_existing_photo_ids(conn, album_id=album.id)
        if len(existing) >= album.count_photos:
            print("  -> Already downloaded, skipping.\n")
            continue

        try:
            metadata_list = download_album(
                client=client,
                user_id=user_id,
                album_id=album.id,
                album_title=album.title,
                event_name=event_name,
                event_year=year,
            )
            insert_images(conn, metadata_list)
            total_downloaded += len(metadata_list)
            print(f"  -> Downloaded {len(metadata_list)} new images.\n")
        except Exception as e:
            print(f"  -> Error: {e}\n")

        # Rate limit: pause between albums
        time.sleep(2)

    conn.close()
    print(f"\nDone! Total new images downloaded: {total_downloaded}")


if __name__ == "__main__":
    main()
