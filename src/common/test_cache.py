import os
import tempfile

from common.cache import Cache, cache_stats


class TestCache:
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path = cache_file.name
        self.cache = Cache(self.cache_path)
        cache_stats.reset()

    def teardown_method(self, method):
        os.remove(self.cache_path)

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

        # Test cache stats
        assert cache_stats.num_queries[self.cache_path] == 2
        assert cache_stats.num_computes[self.cache_path] == 1
        cache_stats.reset()
        assert cache_stats.num_queries[self.cache_path] == 0
        assert cache_stats.num_computes[self.cache_path] == 0
