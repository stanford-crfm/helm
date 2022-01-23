import shutil
import os
from common.general import ensure_file_downloaded, format_tags


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
