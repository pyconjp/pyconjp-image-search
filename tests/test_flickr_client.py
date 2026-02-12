"""Tests for Flickr client utilities."""

from pyconjp_image_search.manager.flickr_client import FlickrPhoto, build_photo_url


def test_build_photo_url_default_size():
    photo = FlickrPhoto(id="53912345678", secret="abc123", server="65535", farm=66, title="Test")
    url = build_photo_url(photo)
    assert url == "https://farm66.staticflickr.com/65535/53912345678_abc123_b.jpg"


def test_build_photo_url_custom_size():
    photo = FlickrPhoto(id="12345", secret="xyz", server="1234", farm=1, title="Test")
    url = build_photo_url(photo, size="z")
    assert url == "https://farm1.staticflickr.com/1234/12345_xyz_z.jpg"
