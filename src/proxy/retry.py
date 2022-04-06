from retrying import Retrying

from common.request import RequestResult
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


# Number of retries
_NUM_RETRIES: int = 5

# Used to calculate the wait between retries (2^r * _WAIT_EXPONENTIAL_MULTIPLIER seconds)
# where r is the number of attempts so far
_WAIT_EXPONENTIAL_MULTIPLIER_SECONDS: int = 3


def _wait(attempts: int, delay: float) -> float:
    """
    Wait function to pass into `Retrying` that logs and returns the amount of time to sleep
    depending on the number of attempts and delay (in milliseconds).
    """
    hlog(f"The request failed. Attempt #{attempts + 1}, retrying in {delay // 1000} seconds...")
    return _retrying.exponential_sleep(attempts, delay)


def _retry_if_request_failed(result: RequestResult) -> bool:
    """Fails if `success` of `RequestResult` is false."""
    return not result.success


# Create a decorator that will retry making requests with exponential backoff
_retrying = Retrying(
    retry_on_result=_retry_if_request_failed,
    wait_func=_wait,
    stop_max_attempt_number=_NUM_RETRIES,
    wait_exponential_multiplier=_WAIT_EXPONENTIAL_MULTIPLIER_SECONDS * 1000,
)

retry_request = lambda f: lambda *args, **kwargs: _retrying.call(f, *args, **kwargs)
