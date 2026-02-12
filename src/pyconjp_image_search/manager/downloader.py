"""Download images from Flickr albums."""

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pyconjp_image_search.config import DATA_DIR
from pyconjp_image_search.manager.flickr_client import (
    FlickrClient,
    FlickrPhoto,
    build_photo_url,
)
from pyconjp_image_search.models import ImageMetadata


def download_album(
    client: FlickrClient,
    user_id: str,
    album_id: str,
    album_title: str,
    event_name: str,
    event_year: int,
    event_type: str = "conference",
    size: str = "b",
    existing_photo_ids: set[str] | None = None,
) -> list[ImageMetadata]:
    """Download all photos from a Flickr album.

    Args:
        client: FlickrClient instance.
        user_id: Flickr user ID.
        album_id: Flickr photoset ID.
        album_title: Human-readable album name (used as directory name).
        event_name: Event name for DB metadata.
        event_year: Event year for DB metadata.
        event_type: Event type for DB metadata.
        size: Flickr size suffix (default "b" = 1024px).
        existing_photo_ids: Photo IDs already in DB (skip these).

    Returns:
        List of ImageMetadata for successfully downloaded images.
    """
    existing = existing_photo_ids or set()

    dir_name = _sanitize_dirname(album_title)
    album_dir = DATA_DIR / dir_name
    album_dir.mkdir(parents=True, exist_ok=True)

    all_photos = client.get_all_photos_in_album(album_id, user_id)

    # Filter out already downloaded and invalid photos (farm=0 means unavailable)
    new_photos = [p for p in all_photos if p.id not in existing and p.farm != 0]

    if not new_photos:
        return []

    metadata_list: list[ImageMetadata] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task(f"Downloading {album_title}", total=len(new_photos))

        with httpx.Client(timeout=120) as http_client:
            for photo in new_photos:
                meta = _download_single_photo(
                    http_client,
                    photo,
                    album_dir,
                    dir_name,
                    album_id,
                    album_title,
                    event_name,
                    event_year,
                    event_type,
                    size,
                )
                if meta is not None:
                    metadata_list.append(meta)
                progress.advance(task)

    return metadata_list


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(httpx.TimeoutException),
)
def _download_single_photo(
    client: httpx.Client,
    photo: FlickrPhoto,
    album_dir: Path,
    dir_name: str,
    album_id: str,
    album_title: str,
    event_name: str,
    event_year: int,
    event_type: str,
    size: str,
) -> ImageMetadata | None:
    """Download a single photo and return its metadata."""
    url = build_photo_url(photo, size)
    filename = f"{photo.id}.jpg"
    local_path = album_dir / filename
    relative_path = f"{dir_name}/{filename}"

    if local_path.exists():
        return None

    try:
        resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError:
        return None

    content = resp.content
    local_path.write_bytes(content)

    try:
        img = Image.open(BytesIO(content))
        image_format = img.format
        width, height = img.size
    except Exception:
        image_format = None
        width = None
        height = None

    return ImageMetadata(
        id=None,
        image_url=url,
        relative_path=relative_path,
        local_filename=filename,
        flickr_photo_id=photo.id,
        album_id=album_id,
        album_title=album_title,
        event_name=event_name,
        event_year=event_year,
        event_type=event_type,
        image_format=image_format,
        width=width,
        height=height,
        file_size_bytes=len(content),
        downloaded_at=datetime.now(UTC),
        created_at=None,
    )


def _sanitize_dirname(name: str) -> str:
    """Convert album title to a safe directory name."""
    safe = "".join(c if c.isalnum() or c in "-_ " else "" for c in name)
    return safe.strip().replace(" ", "_").lower()
