import shutil
import os
from common.general import ensure_file_downloaded


def test_ensure_file_downloaded():
    ensure_file_downloaded("https://ftp.gnu.org/gnu/tar/tar-1.34.tar.gz", "test-tar", untar=True)
    assert os.path.isdir("test-tar")
    shutil.rmtree("test-tar")
