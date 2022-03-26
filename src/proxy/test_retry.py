from retrying import RetryError

from common.request import RequestResult
from .retry import retry_request


def test_retry_for_successful_request():
    @retry_request
    def make_request() -> RequestResult:
        return RequestResult(success=True, completions=[], cached=False)

    result: RequestResult = make_request()
    assert result.success


def test_retry_for_failed_request():
    @retry_request
    def make_request() -> RequestResult:
        return RequestResult(success=False, completions=[], cached=False)

    # Should throw a `RetryError`
    try:
        make_request()
        assert False
    except Exception as e:
        assert isinstance(e, RetryError)
        result: RequestResult = e.last_attempt.value
        assert not result.success
