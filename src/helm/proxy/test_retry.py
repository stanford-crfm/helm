from retrying import RetryError

from helm.common.request import RequestResult
from helm.proxy.retry import retry_request, get_retry_decorator, retry_if_request_failed


def test_retry_for_successful_request():
    @retry_request
    def make_request() -> RequestResult:
        return RequestResult(success=True, completions=[], cached=False, embedding=[])

    result: RequestResult = make_request()
    assert result.success


def test_retry_for_failed_request():
    retry_fail_fast = get_retry_decorator(
        operation="Request",
        max_attempts=1,
        wait_exponential_multiplier_seconds=1,
        retry_on_result=retry_if_request_failed,
    )

    @retry_fail_fast
    def make_request() -> RequestResult:
        return RequestResult(success=False, completions=[], cached=False, embedding=[])

    # Should throw a `RetryError`
    try:
        make_request()
        assert False
    except Exception as e:
        assert isinstance(e, RetryError)
        result: RequestResult = e.last_attempt.value
        assert not result.success
