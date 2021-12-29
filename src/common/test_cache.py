import os
import tempfile
import unittest

from sqlitedict import SqliteDict

from common.cache import Cache


class CacheTest(unittest.TestCase):
    def setUp(self):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path = cache_file.name
        self.cache = Cache(self.cache_path)

    def tearDown(self):
        os.remove(self.cache_path)

    def test_read(self):
        with SqliteDict(self.cache_path) as d:
            d["key1"] = "response1"
            d.commit()

        self.cache.read()
        self.assertTrue("key1" in self.cache.data)
        self.assertEqual(self.cache.data["key1"], "response1")

    def test_write(self):
        self.cache.data["key1"] = "response1"
        self.cache.write()

        with SqliteDict(self.cache_path) as d:
            self.assertTrue("key1" in d)
            self.assertEqual(d["key1"], "response1")

    def test_get(self):
        request = {"name": "request1"}
        compute = lambda: {"response1"}

        # The request should not be cached the first time
        response, cached = self.cache.get(request, compute)
        self.assertEqual(response, {"response1"})
        self.assertFalse(cached)

        # The same request should now be cached
        response, cached = self.cache.get(request, compute)
        self.assertEqual(response, {"response1"})
        self.assertTrue(cached)
