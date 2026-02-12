"""Tests for downloader utility functions."""

from pyconjp_image_search.manager.downloader import _sanitize_dirname


def test_sanitize_dirname_basic():
    assert _sanitize_dirname("PyCon JP 2024") == "pycon_jp_2024"


def test_sanitize_dirname_special_chars():
    assert _sanitize_dirname("PyCon JP 2024 (Day 1)") == "pycon_jp_2024_day_1"


def test_sanitize_dirname_japanese():
    # Japanese characters are alphanumeric (isalnum() returns True), so they are kept
    assert _sanitize_dirname("PyCon mini 東海 2025") == "pycon_mini_東海_2025"


def test_sanitize_dirname_hyphens_underscores():
    assert _sanitize_dirname("my-album_name") == "my-album_name"


def test_sanitize_dirname_extra_spaces():
    assert _sanitize_dirname("  spaces  everywhere  ") == "spaces__everywhere"
