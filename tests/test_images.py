from datetime import datetime
from pathlib import Path

from selfies_for_a_year.images import _filename_date, sort_images


def test_filename_date_iso8601():
    path = Path("2026-01-02T07:19:37-08:00.jpg")
    assert _filename_date(path) == datetime(2026, 1, 2, 7, 19, 37)


def test_filename_date_macos_screenshot():
    path = Path("2024-01-02 at 10.28.21.png")
    assert _filename_date(path) == datetime(2024, 1, 2, 10, 28, 21)


def test_filename_date_date_only():
    path = Path("2023-01-05_1.heic")
    assert _filename_date(path) == datetime(2023, 1, 5)


def test_filename_date_no_match():
    assert _filename_date(Path("IMG_1234.jpg")) is None


def test_sort_images_orders_chronologically(tmp_path):
    names = ["2023-06-01.jpg", "2021-03-05.jpg", "2022-12-31.jpg"]
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"")
        paths.append(p)

    ordered = sort_images(paths)

    assert [p.name for p in ordered] == [
        "2021-03-05.jpg",
        "2022-12-31.jpg",
        "2023-06-01.jpg",
    ]
