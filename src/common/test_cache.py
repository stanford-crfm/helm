import os
import tempfile
import unittest

from common.cache import Cache, CacheConfig, cache_stats

from sqlitedict import SqliteDict


class TestCache(unittest.TestCase):
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path = cache_file.name
        self.cache = Cache(CacheConfig(self.cache_path))
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

    def test_follower(self):
        request_1 = {"name": "request1"}
        compute_1 = lambda: {"response1"}

        response, cached = self.cache.get(request_1, compute_1)
        assert response == {"response1"}
        assert not cached
        assert cache_stats.num_queries[self.cache_path] == 1
        assert cache_stats.num_computes[self.cache_path] == 1

        follower_cache_file = tempfile.NamedTemporaryFile(delete=False)
        follower_cache_path = follower_cache_file.name
        with follower_cache_file:
            cache_with_follower_config = CacheConfig(
                cache_path=self.cache_path, follower_cache_path=follower_cache_path
            )
            cache_with_follower = Cache(cache_with_follower_config)

            response, cached = cache_with_follower.get(request_1, compute_1)
            assert response == {"response1"}
            assert cached
            assert cache_stats.num_queries[self.cache_path] == 2
            assert cache_stats.num_computes[self.cache_path] == 1
            assert cache_stats.num_queries[follower_cache_path] == 0
            assert cache_stats.num_computes[follower_cache_path] == 0

            request_2 = {"name": "request2"}
            compute_2 = lambda: {"response2"}

            response, cached = cache_with_follower.get(request_2, compute_2)
            assert response == {"response2"}
            assert not cached
            assert cache_stats.num_queries[self.cache_path] == 3
            assert cache_stats.num_computes[self.cache_path] == 2
            assert cache_stats.num_queries[follower_cache_path] == 0
            assert cache_stats.num_computes[follower_cache_path] == 0

            expected_dict = {
                '{"name": "request1"}': {"response1"},
                '{"name": "request2"}': {"response2"},
            }
            self.assertCountEqual(SqliteDict(follower_cache_path).items(), expected_dict.items())
