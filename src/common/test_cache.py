import tempfile
import threading

from common.cache import Cache, cache_stats


class TestCache:
    def setup_method(self, method):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_path = self.temp_dir.name
        cache_stats.reset()

    def teardown_method(self, method):
        self.temp_dir.cleanup()

    def test_get(self):
        cache = Cache(self.cache_path)
        request = {"name": "request1"}
        compute = lambda: {"response1"}

        # The request should not be cached the first time
        response, cached = cache.get(request, compute)
        assert response == {"response1"}
        assert not cached

        # The same request should now be cached
        response, cached = cache.get(request, compute)
        assert response == {"response1"}
        assert cached

        # Test cache stats
        assert cache_stats.num_queries[self.cache_path] == 2
        assert cache_stats.num_computes[self.cache_path] == 1
        cache_stats.reset()
        assert cache_stats.num_queries[self.cache_path] == 0
        assert cache_stats.num_computes[self.cache_path] == 0

    def test_many_requests(self):
        cache = Cache(self.cache_path)
        num_items = 100
        num_iterations = 20
        requests = [{"name": "request%d" % i} for i in range(num_items)]
        responses = ["response%d" % i for i in range(num_items)]
        for iteration in range(num_iterations):
            for i in range(num_items):
                response, cached = cache.get(requests[i], lambda: {responses[i]})
                assert response == {responses[i]}
                assert cached == (iteration > 0)
        assert cache_stats.num_queries[self.cache_path] == num_items * num_iterations
        assert cache_stats.num_computes[self.cache_path] == num_items

    def test_many_caches(self):
        num_items = 100
        num_caches = 20
        requests = [{"name": "request%d" % i} for i in range(num_items)]
        responses = ["response%d" % i for i in range(num_items)]
        for cache_index in range(num_caches):
            cache = Cache(self.cache_path)
            for i in range(num_items):
                response, cached = cache.get(requests[i], lambda: {responses[i]})
                assert response == {responses[i]}
                assert cached == (cache_index > 0)
        assert cache_stats.num_queries[self.cache_path] == num_items * num_caches
        assert cache_stats.num_computes[self.cache_path] == num_items

    def test_many_threads(self):
        cache = Cache(self.cache_path)
        num_items = 100
        num_threads = 20
        requests = [{"name": "request%d" % i} for i in range(num_items)]
        responses = ["response%d" % i for i in range(num_items)]

        def run():
            for i in range(num_items):
                response, _ = cache.get(requests[i], lambda: {responses[i]})
                assert response == {responses[i]}

        threads = [threading.Thread(target=run) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert cache_stats.num_queries[self.cache_path] == num_items * num_threads
        assert cache_stats.num_computes[self.cache_path] >= num_items
        assert cache_stats.num_computes[self.cache_path] <= num_items * num_threads
