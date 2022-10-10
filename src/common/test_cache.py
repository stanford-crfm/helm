import os
import tempfile
import unittest
import threading

from common.cache import Cache, CacheConfig, cache_stats

from sqlitedict import SqliteDict


class TestCache(unittest.TestCase):
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path = cache_file.name
        cache_stats.reset()

    def teardown_method(self, method):
        os.remove(self.cache_path)

    def test_get(self):
        cache = Cache(CacheConfig(self.cache_path))

        request = {"name": "request1"}
        compute = lambda: {"response": "response1"}

        # The request should not be cached the first time
        response, cached = cache.get(request, compute)
        assert response == {"response": "response1"}
        assert not cached

        # The same request should now be cached
        response, cached = cache.get(request, compute)
        assert response == {"response": "response1"}
        assert cached

        # Test cache stats
        assert cache_stats.num_queries[self.cache_path] == 2
        assert cache_stats.num_computes[self.cache_path] == 1
        cache_stats.reset()
        assert cache_stats.num_queries[self.cache_path] == 0
        assert cache_stats.num_computes[self.cache_path] == 0

    def test_many_requests(self):
        cache = Cache(CacheConfig(self.cache_path))
        num_items = 10  # TODO: Inrcrease to 100
        num_iterations = 5  # TODO: Inrcrease to 20
        requests = [{"name": "request%d" % i} for i in range(num_items)]
        responses = [{"response": "response%d" % i} for i in range(num_items)]
        for iteration in range(num_iterations):
            for i in range(num_items):
                response, cached = cache.get(requests[i], lambda: responses[i])
                assert response == responses[i]
                assert cached == (iteration > 0)
        assert cache_stats.num_queries[self.cache_path] == num_items * num_iterations
        assert cache_stats.num_computes[self.cache_path] == num_items

    def test_many_caches(self):
        num_items = 10  # TODO: Inrcrease to 100
        num_caches = 5  # TODO: Inrcrease to 20
        requests = [{"name": "request%d" % i} for i in range(num_items)]
        responses = [{"response": "response%d" % i} for i in range(num_items)]
        for cache_index in range(num_caches):
            cache = Cache(CacheConfig(self.cache_path))
            for i in range(num_items):
                response, cached = cache.get(requests[i], lambda: responses[i])
                assert response == responses[i]
                assert cached == (cache_index > 0)
        assert cache_stats.num_queries[self.cache_path] == num_items * num_caches
        assert cache_stats.num_computes[self.cache_path] == num_items

    def test_many_threads(self):
        cache = Cache(CacheConfig(self.cache_path))
        num_items = 10  # TODO: Inrcrease to 100
        num_threads = 5  # TODO: Inrcrease to 20
        requests = [{"name": "request%d" % i} for i in range(num_items)]
        responses = [{"response": "response%d" % i} for i in range(num_items)]

        def run():
            for i in range(num_items):
                response, _ = cache.get(requests[i], lambda: responses[i])
                assert response == responses[i]

        threads = [threading.Thread(target=run) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert cache_stats.num_queries[self.cache_path] == num_items * num_threads
        assert cache_stats.num_computes[self.cache_path] >= num_items
        assert cache_stats.num_computes[self.cache_path] <= num_items * num_threads

    def test_follower(self):
        cache = Cache(CacheConfig(self.cache_path))
        request_1 = {"name": "request1"}
        compute_1 = lambda: {"response": "response1"}

        response, cached = cache.get(request_1, compute_1)
        assert response == {"response": "response1"}
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
            assert response == {"response": "response1"}
            assert cached
            assert cache_stats.num_queries[self.cache_path] == 2
            assert cache_stats.num_computes[self.cache_path] == 1
            assert cache_stats.num_queries[follower_cache_path] == 0
            assert cache_stats.num_computes[follower_cache_path] == 0

            request_2 = {"name": "request2"}
            compute_2 = lambda: {"response": "response2"}

            response, cached = cache_with_follower.get(request_2, compute_2)
            assert response == {"response": "response2"}
            assert not cached
            assert cache_stats.num_queries[self.cache_path] == 3
            assert cache_stats.num_computes[self.cache_path] == 2
            assert cache_stats.num_queries[follower_cache_path] == 0
            assert cache_stats.num_computes[follower_cache_path] == 0

            expected_dict = {
                '{"name": "request1"}': {"response": "response1"},
                '{"name": "request2"}': {"response": "response2"},
            }
            self.assertCountEqual(SqliteDict(follower_cache_path).items(), expected_dict.items())
