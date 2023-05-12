from typing import Callable, Union

from retrying import Retrying

from helm.common.request import RequestResult
from helm.common.tokenization_request import TokenizationRequestResult
from helm.common.hierarchical_logger import hlog, htrack_block
import traceback
import threading

"""
Decorator that will retry after a failure with exponential backoff.
Throws a `RetryError` when the limit of retries is reached.

Example usage:

    @retry_request
    def make_request(request: Request) -> RequestResult:
        ...
"""

# The lock is used to prevent multiple threads from printing at the same time.
# This can cause issues when printing the stack trace.
# (The stack traces can get mixed up and become unreadable.)
print_lock: threading.Lock = threading.Lock()


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
        hlog(
            f"{operation} failed. Retrying (attempt #{attempts + 1}) in {delay // 1000} seconds... "
            "(See above for details)"
        )
        return _retrying.exponential_sleep(attempts, delay)

    def stops_on_keyboard_interrupt(exception: Exception) -> bool:
        """
        This function always return True, as the exception should always be retried.
        It is used to stop retrying on keyboard interrupt.
        """
        return isinstance(exception, KeyboardInterrupt)

    def print_exception_and_traceback(exception: Exception) -> bool:
        """
        This function always return True, as the exception should always be retried.
        It is used to print the exception and traceback (HACK).
        TODO: Should not retry on keyboard interrupt. (right now it is inconsistent)
        """
        with print_lock:
            hlog("")
            hlog_exception(exception)
        return not stops_on_keyboard_interrupt(exception)

    _retrying = Retrying(
        retry_on_result=retry_on_result,
        wait_func=wait,
        stop_max_attempt_number=max_attempts,
        # Used to calculate the wait between retries (2^r * wait_exponential_multiplier_seconds seconds)
        wait_exponential_multiplier=wait_exponential_multiplier_seconds * 1000,
        retry_on_exception=print_exception_and_traceback,
        wrap_exception=True,
    )

    return lambda f: lambda *args, **kwargs: _retrying.call(f, *args, **kwargs)


def hlog_exception(exception: Exception):
    with htrack_block("Exception:", print_time=False):
        hlog(exception)
    with htrack_block("Traceback:", print_time=False):
        hlog_traceback_of_exception(exception)


def hlog_traceback_of_exception(exception: Exception):
    traceback_list = traceback.extract_tb(exception.__traceback__)
    for item in traceback.StackSummary.from_list(traceback_list).format():
        items = item.split("\n")
        with htrack_block(items[0], print_time=False):
            hlog(items[1])


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
