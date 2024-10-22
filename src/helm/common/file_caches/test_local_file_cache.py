import os
import shutil
import tempfile
import unittest

from helm.common.file_caches.local_file_cache import LocalFileCache


class TestLocalFileCache(unittest.TestCase):
    def setup_method(self, _):
        self.path: str = tempfile.mkdtemp()

    def teardown_method(self, _):
        shutil.rmtree(self.path)

    def test_get(self):
        cache = LocalFileCache(self.path, file_extension="txt")
        file_path1: str = cache.store(lambda: "hello.".encode())

        # Verify the contents of the file
        with open(file_path1, "r") as f:
            assert f.read() == "hello."

        cache.store(lambda: "bye.".encode())
        assert len(os.listdir(self.path)) == 2
