from typing import Callable, Union

from retrying import Retrying

from common.request import RequestResult
from common.tokenization_request import TokenizationRequestResult
from common.hierarchical_logger import hlog

"""
Decorator that will retry making requests after a failure with exponential backoff.
The retry is triggered when `success` of `RequestResult` is false.
Throws a `RetryError` when the requests fails `_NUM_RETRIES` number of times.

Usage:

    @retry_request
    def make_request(request: Request) -> RequestResult:
        ...
"""

# Defaults for the benchmarking proxy server
# Number of retries
_NUM_RETRIES: int = 5

# Used to calculate the wait between retries (2^r * _WAIT_EXPONENTIAL_MULTIPLIER seconds)
# where r is the number of attempts so far
_WAIT_EXPONENTIAL_MULTIPLIER_SECONDS: int = 3


def get_retry_decorator(stop_max_attempt_number: int, wait_exponential_multiplier_seconds: int) -> Callable:
    """
    Create a decorator that will retry with exponential backoff.
    """

    def wait(attempts: int, delay: float) -> float:
        """
        Wait function to pass into `Retrying` that logs and returns the amount of time to sleep
        depending on the number of attempts and delay (in milliseconds).
        """
        hlog(f"The request failed. Attempt #{attempts + 1}, retrying in {delay // 1000} seconds...")
        return _retrying.exponential_sleep(attempts, delay)

    def retry_if_request_failed(result: Union[RequestResult, TokenizationRequestResult]) -> bool:
        """Fails if `success` of `RequestResult` or `TokenizationRequestResult` is false."""
        return not result.success

    _retrying = Retrying(
        retry_on_result=retry_if_request_failed,
        wait_func=wait,
        stop_max_attempt_number=stop_max_attempt_number,
        wait_exponential_multiplier=wait_exponential_multiplier_seconds * 1000,
    )

    return lambda f: lambda *args, **kwargs: _retrying.call(f, *args, **kwargs)


retry_request: Callable = get_retry_decorator(_NUM_RETRIES, _WAIT_EXPONENTIAL_MULTIPLIER_SECONDS)
