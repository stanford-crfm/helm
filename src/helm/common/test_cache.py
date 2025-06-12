import os
import tempfile
import unittest
import threading

from helm.common.cache import Cache, SqliteCacheConfig, cache_stats, get_all_from_sqlite


class TestCache(unittest.TestCase):
    def setup_method(self, method):
        cache_file = tempfile.NamedTemporaryFile(delete=False)
        self.cache_path = cache_file.name
        cache_stats.reset()

    def teardown_method(self, method):
        os.remove(self.cache_path)

    def test_get(self):
        cache = Cache(SqliteCacheConfig(self.cache_path))

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

    def test_raise(self):
        cache = Cache(SqliteCacheConfig(self.cache_path))
        request = {"name": "request1"}

        def compute():
            raise ValueError("test error")

        with self.assertRaisesRegex(ValueError, "test error"):
            cache.get(request, compute)

    def test_many_requests(self):
        cache = Cache(SqliteCacheConfig(self.cache_path))
        num_items = 10  # TODO: Inrcrease to 100
        num_iterations = 5  # TODO: Inrcrease to 20
        requests = [{"name": f"request{i}"} for i in range(num_items)]
        responses = [{"response": f"response{i}"} for i in range(num_items)]
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
        requests = [{"name": f"request{i}"} for i in range(num_items)]
        responses = [{"response": f"response{i}"} for i in range(num_items)]
        for cache_index in range(num_caches):
            cache = Cache(SqliteCacheConfig(self.cache_path))
            for i in range(num_items):
                response, cached = cache.get(requests[i], lambda: responses[i])
                assert response == responses[i]
                assert cached == (cache_index > 0)
        assert cache_stats.num_queries[self.cache_path] == num_items * num_caches
        assert cache_stats.num_computes[self.cache_path] == num_items

    def test_many_threads(self):
        cache = Cache(SqliteCacheConfig(self.cache_path))
        num_items = 10  # TODO: Inrcrease to 100
        num_threads = 5  # TODO: Inrcrease to 20
        requests = [{"name": f"request{i}"} for i in range(num_items)]
        responses = [{"response": f"response{i}"} for i in range(num_items)]

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

    def test_get_all_from_sqlite(self):
        cache = Cache(SqliteCacheConfig(self.cache_path))
        num_items = 10  # TODO: Inrcrease to 100
        requests = [{"name": f"request{i}"} for i in range(num_items)]
        responses = [{"response": f"response{i}"} for i in range(num_items)]
        for i in range(num_items):
            response, cached = cache.get(requests[i], lambda: responses[i])
            assert response == responses[i]
            assert not cached

        actual_requests = []
        actual_responses = []
        for request, response in get_all_from_sqlite(self.cache_path):
            actual_requests.append(request)
            actual_responses.append(response)
        self.assertCountEqual(actual_requests, requests)
        self.assertCountEqual(actual_responses, responses)
