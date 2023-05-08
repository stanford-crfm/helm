from typing import Callable, Union

from retrying import Retrying

from helm.common.request import RequestResult
from helm.common.tokenization_request import TokenizationRequestResult
from helm.common.hierarchical_logger import hlog

"""
Decorator that will retry after a failure with exponential backoff.
Throws a `RetryError` when the limit of retries is reached.

Example usage:

    @retry_request
    def make_request(request: Request) -> RequestResult:
        ...
"""


def get_retry_decorator(
    operation: str, max_attempts: int, wait_exponential_multiplier_seconds: int, retry_on_result: Callable
) -> Callable:
    """
    Create a decorator that will retry with exponential backoff.
    """

    def wait(attempts: int, delay: float) -> float:
        """
        Wait function to pass into `Retrying` that logs and returns the amount of time to sleep
        depending on the number of attempts and delay (in milliseconds).
        """
        hlog(f"{operation} failed. Retrying (attempt #{attempts + 1}) in {delay // 1000} seconds...")
        return _retrying.exponential_sleep(attempts, delay)

    _retrying = Retrying(
        retry_on_result=retry_on_result,
        wait_func=wait,
        stop_max_attempt_number=max_attempts,
        # Used to calculate the wait between retries (2^r * wait_exponential_multiplier_seconds seconds)
        wait_exponential_multiplier=wait_exponential_multiplier_seconds * 1000,
    )

    return lambda f: lambda *args, **kwargs: _retrying.call(f, *args, **kwargs)


def retry_if_request_failed(result: Union[RequestResult, TokenizationRequestResult]) -> bool:
    """Fails if `success` of `RequestResult` or `TokenizationRequestResult` is false."""
    if not result.success:
        hlog(result.error)
    retry_if_fail: bool = True
    if isinstance(result, RequestResult):
        retry_if_fail = (
            result.error_flags is None or result.error_flags.is_retriable is None or result.error_flags.is_retriable
        )
    return not result.success and retry_if_fail


retry_request: Callable = get_retry_decorator(
    "Request", max_attempts=8, wait_exponential_multiplier_seconds=5, retry_on_result=retry_if_request_failed
)
