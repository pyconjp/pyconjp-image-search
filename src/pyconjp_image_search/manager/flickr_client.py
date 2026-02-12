"""Flickr REST API client."""

import time
from dataclasses import dataclass

import httpx

from pyconjp_image_search.config import FLICKR_API_BASE, FLICKR_API_KEY


@dataclass(frozen=True)
class FlickrPhoto:
    """Metadata for a single photo from Flickr API."""

    id: str
    secret: str
    server: str
    farm: int
    title: str


@dataclass(frozen=True)
class FlickrAlbum:
    """Metadata for a Flickr album/photoset."""

    id: str
    title: str
    description: str
    count_photos: int


class FlickrClient:
    """Client for Flickr REST API."""

    def __init__(self, api_key: str | None = None, timeout: int = 30) -> None:
        self.api_key = api_key or FLICKR_API_KEY
        if not self.api_key:
            raise ValueError("Flickr API key is required. Set FLICKR_API_KEY in .env file.")
        self.timeout = timeout

    def _call(self, method: str, **params: str) -> dict:
        """Make a Flickr API call and return the parsed JSON."""
        params.update(
            {
                "method": method,
                "api_key": self.api_key,
                "format": "json",
                "nojsoncallback": "1",
            }
        )
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(FLICKR_API_BASE, params=params)
            resp.raise_for_status()
        data = resp.json()
        if data.get("stat") != "ok":
            raise RuntimeError(f"Flickr API error: {data}")
        return data

    def list_albums(self, user_id: str) -> list[FlickrAlbum]:
        """List all albums for a user."""
        albums: list[FlickrAlbum] = []
        page = 1
        while True:
            data = self._call(
                "flickr.photosets.getList",
                user_id=user_id,
                page=str(page),
                per_page="500",
            )
            for ps in data["photosets"]["photoset"]:
                albums.append(
                    FlickrAlbum(
                        id=ps["id"],
                        title=ps["title"]["_content"],
                        description=ps["description"]["_content"],
                        count_photos=int(ps["photos"]),
                    )
                )
            if page >= int(data["photosets"]["pages"]):
                break
            page += 1
            time.sleep(0.1)
        return albums

    def get_all_photos_in_album(self, album_id: str, user_id: str) -> list[FlickrPhoto]:
        """Fetch all photos across all pages of an album."""
        all_photos: list[FlickrPhoto] = []
        page = 1
        while True:
            data = self._call(
                "flickr.photosets.getPhotos",
                photoset_id=album_id,
                user_id=user_id,
                page=str(page),
                per_page="500",
            )
            photoset = data["photoset"]
            for p in photoset["photo"]:
                all_photos.append(
                    FlickrPhoto(
                        id=p["id"],
                        secret=p["secret"],
                        server=p["server"],
                        farm=int(p["farm"]),
                        title=p["title"],
                    )
                )
            if page >= int(photoset["pages"]):
                break
            page += 1
            time.sleep(0.1)
        return all_photos


def build_photo_url(photo: FlickrPhoto, size: str = "b") -> str:
    """Build the static Flickr photo URL.

    Size suffixes: s=75sq, q=150sq, t=100, m=240, z=640,
    b=1024, h=1600, k=2048, o=original
    """
    return (
        f"https://farm{photo.farm}.staticflickr.com/"
        f"{photo.server}/{photo.id}_{photo.secret}_{size}.jpg"
    )
