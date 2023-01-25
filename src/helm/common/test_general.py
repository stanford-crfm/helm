import shutil
import os
from helm.common.general import (
    ensure_file_downloaded,
    format_tags,
    flatten_list,
    format_split,
    get_file_name,
    unique_simplification,
    is_url,
)


def test_ensure_file_downloaded():
    ensure_file_downloaded("https://ftp.gnu.org/gnu/tar/tar-1.34.tar.gz", "test-tar", unpack=True, unpack_type="untar")
    assert os.path.isdir("test-tar")
    shutil.rmtree("test-tar")

    ensure_file_downloaded("https://ftp.gnu.org/gnu/gzip/gzip-1.9.zip", "test-zip", unpack=True)
    assert os.path.isdir("test-zip")
    shutil.rmtree("test-zip")


def test_format_tags():
    tags = ["tag_1", "tag_2", "tag_3"]
    assert format_tags(tags) == "[tag_1,tag_2,tag_3]"


def test_flatten_list():
    nested_lists = [[1, 2], [3], [4, [5, 6]]]
    assert flatten_list(nested_lists) == [1, 2, 3, 4, 5, 6]


def test_format_split():
    assert format_split("split") == "|split|"


def test_unique_simplification():
    assert unique_simplification([{"model": "A"}, {"model": "B"}], []) == [{"model": "A"}, {"model": "B"}]

    # model differs, remove n
    assert unique_simplification([{"model": "A", "n": 3}, {"model": "B", "n": 3}], []) == [
        {"model": "A"},
        {"model": "B"},
    ]

    # model and n differ, remove n
    assert unique_simplification([{"model": "A", "n": 3}, {"model": "B", "n": 4}], ["model"]) == [
        {"model": "A"},
        {"model": "B"},
    ]

    # n differs, remove model
    assert unique_simplification([{"model": "A", "n": 3}, {"model": "A", "n": 4}], ["model"]) == [
        {"n": 3},
        {"n": 4},
    ]


def test_get_file_name():
    assert get_file_name("/path/to/image.png") == "image.png"


def test_is_url():
    assert is_url("https://crfm.stanford.edu")
    assert not is_url("/some/path")
