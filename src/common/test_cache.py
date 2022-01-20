import os
import tempfile

from sqlitedict import SqliteDict

from common.cache import Cache


class TestCache:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path = cache_file.name
        self.cache = Cache(self.cache_path)

    def teardown_method(self, method):
        os.remove(self.cache_path)

    def test_read(self):
        with SqliteDict(self.cache_path) as d:
            d["key1"] = "response1"
            d.commit()

        self.cache.read()
        assert "key1" in self.cache.data
        assert self.cache.data["key1"] == "response1"

    def test_write(self):
        self.cache.data["key1"] = "response1"
        self.cache.write()

        with SqliteDict(self.cache_path) as d:
            assert "key1" in d
            assert d["key1"] == "response1"

    def test_get(self):
        request = {"name": "request1"}
        compute = lambda: {"response1"}

        # The request should not be cached the first time
        response, cached = self.cache.get(request, compute)
        assert response == {"response1"}
        assert not cached

        # The same request should now be cached
        response, cached = self.cache.get(request, compute)
        assert response == {"response1"}
        assert cached
