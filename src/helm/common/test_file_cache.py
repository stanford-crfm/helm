import os
import shutil
import tempfile
import unittest

from helm.common.file_cache import FileCache


class TestFileCache(unittest.TestCase):
    def setup_method(self, _):
        self.path: str = tempfile.mkdtemp()

    def teardown_method(self, _):
        shutil.rmtree(self.path)

    def test_get(self):
        cache = FileCache(self.path, file_extension="text")
        file_path1: str = cache.store(lambda: b"hello.")

        # Verify the contents of the file
        with open(file_path1, "r") as f:
            assert f.read() == "hello."

        cache.store(lambda: b"bye.")
        assert len(os.listdir(self.path)) == 2
